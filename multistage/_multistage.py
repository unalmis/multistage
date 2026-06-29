"""General multi-stage neural network classes."""

import os
import time
import warnings
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import optax
import optimistix
import orbax.checkpoint as ocp
from jax import config, jit, value_and_grad
from paramax import non_trainable, unwrap

from ._io_utils import _ParamContainer, checkpoint_manager, save
from ._utils import (
    _split_pde_loss_components,
    adaptive_sample,
    is_not_trainable,
    make_weighted_pde_loss,
    make_weighted_pde_residual_loss,
    partition,
    rescale,
    stats,
    stats_chebyshev,
    weighted_pde_loss,
)

config.update("jax_enable_x64", True)


def _coerce_params(params, params_are_trainable):
    """Return a consistently initialized parameter container."""
    if params is None:
        return _ParamContainer({})
    if not isinstance(params, _ParamContainer):
        params = _ParamContainer(params)
    if params_are_trainable:
        return params
    return _ParamContainer(
        {
            key: val if (val is None or is_not_trainable(val)) else non_trainable(val)
            for key, val in params.items()
        }
    )


def _trainable_params_or_none(params):
    """Extract trainable PDE parameters, returning None when there are none."""
    if params is None:
        return None
    params = eqx.filter(
        params, is_not_trainable, inverse=True, is_leaf=is_not_trainable
    )
    leaves = [
        leaf for leaf in jax.tree_util.tree_leaves(params) if eqx.is_inexact_array(leaf)
    ]
    return params if leaves else None


_DEFAULT_STAGE_CORRECTION_PARAM_MAP = {"log_lambda_2": "lambda_2"}
_TRANSFORMED_PARAM_PREFIXES = ("log_", "exp_")


def _stage_correction_entry(key, val, correction_param_map):
    spec = correction_param_map.get(key, key)
    if callable(spec):
        correction_key, correction_val = spec(key, val)
    elif isinstance(spec, tuple):
        correction_key, initializer = spec
        correction_val = initializer(val) if callable(initializer) else initializer
    else:
        correction_key = spec
        correction_val = jnp.zeros_like(val)
    return correction_key, correction_val


def _stage_correction_params_or_none(params, correction_param_map=None):
    """Initialize trainable PDE parameter corrections for the next stage.

    A later stage stores corrections, not another copy of the previous total
    estimate. Transformed parameters require an explicit map because their
    neutral correction value depends on how the PDE residual combines stages.
    """
    correction_param_map = {
        **_DEFAULT_STAGE_CORRECTION_PARAM_MAP,
        **({} if correction_param_map is None else correction_param_map),
    }

    params = _trainable_params_or_none(params)
    if params is None:
        return None

    corrections = {}
    for key, val in params.items():
        if val is None:
            continue
        if key not in correction_param_map and key.startswith(
            _TRANSFORMED_PARAM_PREFIXES
        ):
            raise ValueError(
                f"Parameter '{key}' looks transformed, but no stage correction "
                "initializer was provided for it. Pass correction_param_map to "
                "prescribe the neutral next-stage parameter."
            )
        correction_key, correction_val = _stage_correction_entry(
            key, val, correction_param_map
        )
        if correction_key in corrections:
            raise ValueError(
                "Multiple trainable parameters initialize the same correction "
                f"parameter '{correction_key}'. Check the stage correction map."
            )
        corrections[correction_key] = correction_val

    return _ParamContainer(corrections) if corrections else None


def _feature_scale_from_frequency(kappa, in_size, feature_map="separable"):
    """Convert target angular frequency to an input scale for ``eqx.nn.Linear``.

    Equinox initializes ``Linear(in_size, out_size)`` weights uniformly on
    ``[-1 / sqrt(in_size), 1 / sqrt(in_size)]``, so each first-layer weight has
    RMS magnitude ``1 / sqrt(3 * in_size)``. Separable features mask each row to
    one coordinate, so the masked component needs the full ``sqrt(3 * in_size)``
    compensation. Dense random features use all coordinates in each row; using
    ``sqrt(3)`` keeps an isotropic random wave-vector's RMS norm at ``kappa``
    instead of overscaling it by ``sqrt(in_size)``.
    """
    kappa = jnp.asarray(kappa)
    if feature_map == "separable":
        return kappa * jnp.sqrt(3.0 * in_size)
    if feature_map == "random":
        return kappa * jnp.sqrt(3.0)
    raise ValueError("feature_map must be 'separable' or 'random'.")


def _feature_mask(width_size, in_size, feature_map):
    """Return a first-layer mask for the requested Fourier feature geometry."""
    if feature_map == "random":
        return jnp.ones((width_size, in_size))
    if feature_map == "separable":
        axis = jnp.arange(width_size) % in_size
        return jax.nn.one_hot(axis, in_size)
    raise ValueError("feature_map must be 'separable' or 'random'.")


def _safe_loss_ref(loss_ref):
    """Return a positive finite scalar suitable for loss normalization."""
    loss_ref = jnp.asarray(loss_ref)
    return jnp.where(
        jnp.isfinite(loss_ref) & (loss_ref > 0),
        loss_ref,
        jnp.ones((), dtype=loss_ref.dtype),
    )


def _as_tuple(value):
    """Normalize scalar/list constructor values for multi-correction stages."""
    if isinstance(value, (tuple, list)):
        return tuple(value)
    value = jnp.asarray(value)
    if value.ndim == 0:
        return (value,)
    return tuple(value[i] for i in range(value.shape[0]))


def _as_kappa_tuple(kappas, in_size):
    """Normalize one or more kappa vectors."""
    if isinstance(kappas, (tuple, list)):
        out = tuple(jnp.asarray(kappa) for kappa in kappas)
    else:
        kappas = jnp.asarray(kappas)
        if kappas.ndim == 1:
            out = (kappas,)
        elif kappas.ndim == 2:
            out = tuple(kappas[i] for i in range(kappas.shape[0]))
        else:
            raise ValueError("kappas must have shape (in_size,) or (n, in_size).")

    for kappa in out:
        if kappa.shape != (in_size,):
            raise ValueError("Each kappa must have shape (in_size,).")
    return out


class Stage1(eqx.Module):
    """First stage PINN solver.

    Examples
    --------
      * See ``tests/test_burgers.py``.

    Parameters
    ----------
    lb : jax.Array
        Lower bounds of the domain [x1_min, ..., x_i_min, ..., x_n_min].
    ub : jax.Array
        Upper bounds of the domain [x1_max, ..., x_i_max, ..., x_n_max].
    in_size : int
        Number of dimensions of input.
        The input should to the network should be ``in_size`` arguments.
    out_size : int
        The output should have shape (out_size, )
    width_size : int
        Size of each hidden layer.
    depth : int
        The number of hidden layers, including the output layer.
    activation : callable
        The activation function after each hidden layer.
        Default is ``jnp.tanh``.
    params : dict[str, jax.Array]
        Dictionary of parameters to learn and initial guesses.
        E.g. for Burgers:
        {
            "lambda_1": jax.random.normal(l1_key, (1,)) * 0.1,
            "log_lambda_2": -6.0 + jax.random.normal(l2_key, (1,)) * 0.1,
        }
    params_are_trainable : bool
        Whether the ``params`` values should be frozen or an optimizable quantity.
        Default is False for frozen.
    key : float
        Key for reproducibility.
    kwargs : dict
        Keyword arguments to ``equinox.nn.MLP``.

    """

    _lb: jax.Array
    _ub: jax.Array
    _params: _ParamContainer
    _mlp: eqx.nn.MLP

    def __init__(
        self,
        lb,
        ub,
        in_size,
        out_size,
        width_size=20,
        depth=4,
        activation=jnp.tanh,
        params=None,
        params_are_trainable=False,
        key=None,
        **kwargs,
    ):
        if key is None:
            key = jax.random.PRNGKey(42)

        self._lb = non_trainable(lb)
        self._ub = non_trainable(ub)
        self._mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            activation=activation,
            key=key,
            **kwargs,
        )
        self._params = _coerce_params(params, params_are_trainable)

    @property
    def params(self):
        """Params for this stage."""
        return unwrap(self._params)

    @property
    def lb(self):
        """Lower bound of input coordinates."""
        return unwrap(self._lb)

    @property
    def ub(self):
        """Upper bound of input coordinates."""
        return unwrap(self._ub)

    @property
    def in_size(self):
        """Number of input dimensions."""
        return self._mlp.in_size

    @property
    def out_size(self):
        """Number of output dimensions."""
        return self._mlp.out_size

    @property
    def epsilon(self):
        """Estimated magnitude scale of output."""
        return 1.0

    @property
    def kappa(self):
        """Estimated dominant frequency of output."""
        return jnp.ones(self._mlp.in_size)

    def __call__(self, *args):
        """Compute the output of this network.

        Parameters
        ----------
        args : tuple[jnp.ndarray]
            Input coordinates, e.g. (x, t) for 2D problem.

        """
        x = rescale(jnp.stack(args), self.lb, self.ub)
        x = self._mlp(x)
        if self.out_size == 1:
            x = x.squeeze(0)
        return x

    def get_param(self, key, default=None):
        """Return ``self.params["key"]`` if it exists and is not None else default."""
        val = self.params.get(key, default)
        return default if val is None else val

    def print_params(self):
        """Print the params of this network."""
        print(f"    Stage 1 params: {self.params}")

    def print_frozen_params(self):
        """Print the frozen parameters of this network."""
        pass


class Stage2(eqx.Module):
    """Initializes the Stage 2 PINN model.

    Examples
    --------
      * See ``tests/test_burgers.py``.

    Parameters
    ----------
    s1 : Stage1
        The frozen model from stage 1.
    epsilon : float
        Approximate magnitude of output.
    kappa : jax.Array
        Approximate angular frequency of this stage's output in normalized
        coordinates. For ``feature_map="separable"``, each entry is the target
        frequency for one input direction. For ``feature_map="random"``, the
        vector gives anisotropic component scales; if all entries are equal,
        the random wave-vector RMS norm matches that common value.
        Shape ``(s1.in_size,)``.
    width_size : int
        Size of each hidden layer.
    depth : int
        The number of hidden layers, including the output layer.
    activation : callable
        The activation function for each hidden layer after the first.
        Default is ``jnp.tanh``.
    params : dict[str, jax.Array]
        Dictionary of parameter corrections to learn and initial guesses. The
        automatic multistage constructors initialize corrections so that the
        total PDE parameters are unchanged at stage creation. Manual transformed
        parameters are allowed, but must be initialized in their transformed
        coordinates.
        E.g. for Burgers:
        {
            "lambda_1": jax.random.normal(l1_key, (1,)) * 0.1,
            "log_lambda_2": jnp.log(0.5),
        }
    params_are_trainable : bool
        Whether the ``params`` values should be frozen or an optimizable quantity.
        Default is False for frozen.
    key : float
        Key for reproducibility.
    chebyshev : bool
        Whether the frequency ``kappa`` is associated with a Chebyshev feature
        mapping instead of Fourier. Default is False.
    feature_map : {"separable", "random"}
        First-layer Fourier feature geometry. ``"separable"`` assigns each
        sinusoidal feature to one input coordinate; ``"random"`` preserves the
        original dense random plane-wave mapping and interprets ``kappa`` as an
        isotropic wave-vector norm when all entries are equal.
    kwargs : dict
        Keyword arguments to ``equinox.nn.MLP``.

    """

    _s1: Stage1
    _epsilon: float
    _kappa: jax.Array
    _params: _ParamContainer
    _first: eqx.nn.Linear
    _mlp: eqx.nn.MLP
    _chebyshev: bool
    _feature_map: str
    _feature_mask: jax.Array

    def __init__(
        self,
        s1,
        epsilon,
        kappa,
        width_size=20,
        depth=4,
        activation=jnp.tanh,
        params=None,
        params_are_trainable=False,
        key=None,
        *,
        chebyshev=False,
        feature_map="separable",
        **kwargs,
    ):
        s1 = non_trainable(s1)
        self._s1 = s1
        self._epsilon = non_trainable(epsilon)
        self._kappa = non_trainable(jnp.asarray(kappa))
        self._chebyshev = chebyshev
        self._feature_map = feature_map
        self._feature_mask = non_trainable(
            _feature_mask(width_size, s1.in_size, feature_map)
        )
        if chebyshev:
            warnings.warn("Chebyshev setting is experimental.")

        if key is None:
            key = jax.random.PRNGKey(42)
        key1, key2 = jax.random.split(key, 2)

        self._first = eqx.nn.Linear(s1.in_size, width_size, key=key1)
        self._mlp = eqx.nn.MLP(
            in_size=width_size,
            out_size=s1.out_size,
            width_size=width_size,
            depth=depth - 1,
            activation=activation,
            key=key2,
            **kwargs,
        )
        self._params = _coerce_params(params, params_are_trainable)

    @property
    def s1(self):
        """Returns the previous stage network."""
        return unwrap(self._s1)

    @property
    def epsilon(self):
        """Estimated magnitude scale of output for this stage."""
        return unwrap(self._epsilon)

    @property
    def kappa(self):
        """Estimated dominant frequency of output for this stage."""
        return unwrap(self._kappa)

    @property
    def params(self):
        """Params for this stage."""
        return unwrap(self._params)

    @property
    def in_size(self):
        """Number of input dimensions."""
        return self.s1.in_size

    @property
    def out_size(self):
        """Number of output dimensions."""
        return self._mlp.out_size

    @property
    def lb(self):
        """Lower bound of input coordinates."""
        return self.s1.lb

    @property
    def ub(self):
        """Upper bound of input coordinates."""
        return self.s1.ub

    def __call__(self, *args):
        """Compute the output of this network.

        output = last stage + epsilon * this stage

        Parameters
        ----------
        args : tuple[jnp.ndarray]
            Input coordinates, e.g. (x, t) for 2D problem.

        """
        return self.s1(*args) + self.epsilon * self.compute_s2(*args)

    def compute_s2(self, *args):
        """Compute just this stage of the output."""
        x = rescale(jnp.stack(args), self.lb, self.ub)
        feature_scale = _feature_scale_from_frequency(
            self.kappa, self.in_size, self._feature_map
        )
        weight = self._first.weight * unwrap(self._feature_mask)
        bias = 0 if self._first.bias is None else self._first.bias

        if self._chebyshev:
            # Ensure x ∈ (-1, 1), i.e. where arccos is differentiable.
            eps = 1 - 1e2 * jnp.finfo(jnp.array(1.0).dtype).eps
            x = jnp.clip(x, -eps, eps)
            x = jnp.cos(weight @ (feature_scale * jnp.arccos(x)) + bias)
        else:
            x = jnp.sin(weight @ (feature_scale * x) + bias)
        x = self._mlp(x)

        if self.out_size == 1:
            x = x.squeeze(0)

        return x

    def get_param(self, key, default=None):
        """Return ``self.params["key"]`` if it exists and is not None else default."""
        val = self.params.get(key, default)
        return default if val is None else val

    def print_params(self):
        """Print the params of this network."""
        print(f"    Current params: {self.params}")

    def print_frozen_params(self):
        """Print the frozen parameters of this network."""
        print(
            f"    Current value for (epsilon, kappa) = ({self.epsilon}, {self.kappa})."
        )
        self.s1.print_params()


class MultiCorrectionStage(eqx.Module):
    """A stage that adds multiple correction networks to one frozen base stage.

    This is useful for forward/inverse PDE solves where the next correction is
    expected to contain more than one scale, for example a high-frequency PDE
    residual correction and a lower-frequency parameter-error correction.
    """

    _corrections: tuple
    _params: _ParamContainer

    def __init__(
        self,
        s1,
        epsilons,
        kappas,
        width_size=20,
        depth=4,
        activation=jnp.tanh,
        params=None,
        params_are_trainable=False,
        key=None,
        *,
        chebyshev=False,
        feature_map="separable",
        **kwargs,
    ):
        epsilons = _as_tuple(epsilons)
        kappas = _as_kappa_tuple(kappas, s1.in_size)
        if len(epsilons) != len(kappas):
            raise ValueError("epsilons and kappas must describe the same count.")
        if len(epsilons) < 2:
            raise ValueError("Use Stage2 for a single correction network.")

        if key is None:
            key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, len(epsilons))
        self._corrections = tuple(
            Stage2(
                s1,
                epsilon=epsilon,
                kappa=kappa,
                width_size=width_size,
                depth=depth,
                activation=activation,
                params=None,
                params_are_trainable=False,
                key=stage_key,
                chebyshev=chebyshev,
                feature_map=feature_map,
                **kwargs,
            )
            for epsilon, kappa, stage_key in zip(epsilons, kappas, keys)
        )
        self._params = _coerce_params(params, params_are_trainable)

    @property
    def s1(self):
        """Return the frozen previous stage network."""
        return self._corrections[0].s1

    @property
    def params(self):
        """Params for this stage."""
        return unwrap(self._params)

    @property
    def epsilon(self):
        """Estimated magnitude scale of the first correction."""
        return self._corrections[0].epsilon

    @property
    def epsilons(self):
        """Estimated magnitude scales for all corrections."""
        return tuple(correction.epsilon for correction in self._corrections)

    @property
    def kappa(self):
        """Estimated dominant frequency of the first correction."""
        return self._corrections[0].kappa

    @property
    def kappas(self):
        """Estimated dominant frequencies for all corrections."""
        return tuple(correction.kappa for correction in self._corrections)

    @property
    def in_size(self):
        """Number of input dimensions."""
        return self.s1.in_size

    @property
    def out_size(self):
        """Number of output dimensions."""
        return self.s1.out_size

    @property
    def lb(self):
        """Lower bound of input coordinates."""
        return self.s1.lb

    @property
    def ub(self):
        """Upper bound of input coordinates."""
        return self.s1.ub

    def __call__(self, *args):
        """Compute the previous stage plus all correction networks."""
        out = self.s1(*args)
        for correction in self._corrections:
            out = out + correction.epsilon * correction.compute_s2(*args)
        return out

    def compute_correction(self, index, *args):
        """Compute one unscaled correction network."""
        return self._corrections[index].compute_s2(*args)

    def compute_s2(self, *args):
        """Compute the first unscaled correction for Stage2-compatible code."""
        return self.compute_correction(0, *args)

    def get_param(self, key, default=None):
        """Return ``self.params["key"]`` if it exists and is not None else default."""
        val = self.params.get(key, default)
        return default if val is None else val

    def print_params(self):
        """Print the params of this network."""
        print(f"    Current params: {self.params}")

    def print_frozen_params(self):
        """Print the frozen parameters of this network."""
        print(
            "    Current values for (epsilon, kappa) = "
            f"{tuple(zip(self.epsilons, self.kappas))}."
        )
        self.s1.print_params()


def _is_multiple_or_last(step, multiple, last):
    if multiple <= 0:
        return step == (last - 1)
    return ((step % multiple) == 0) or (step == (last - 1))


def _is_completed_multiple_or_last(step, multiple, last):
    if multiple <= 0:
        return step == (last - 1)
    return (((step + 1) % multiple) == 0) or (step == (last - 1))


def _fill_lask_k_buffer(last_k_loss, loss_history):
    n_restore = min(loss_history.size, last_k_loss.size)
    if n_restore > 0:
        steps = np.arange(loss_history.size - n_restore, loss_history.size)
        last_k_loss[steps % last_k_loss.size] = loss_history[-n_restore:]
    return last_k_loss


def _merge_adaptive_samples(current, new, accumulate=False, max_samples=None):
    """Return the active adaptive samples for the next optimizer step."""
    if not accumulate or current[0] is None:
        merged = new
    else:
        merged = [jnp.concatenate((old, new_i)) for old, new_i in zip(current, new)]

    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_adaptive_samples must be positive or None.")
        merged = [samples[-max_samples:] for samples in merged]
    return merged


def _checkpoint_stage_path(checkpoint_dir, name):
    """Return a stage checkpoint path or None when checkpointing is disabled."""
    if checkpoint_dir is None:
        return None
    return os.path.join(checkpoint_dir, name)


def _expanded_num_samples(num_samples, in_size):
    if len(num_samples) == 1:
        return (num_samples[0],) * in_size
    return tuple(num_samples)


def _warn_if_frequency_underresolved(
    kappa, num_samples, samples_per_mode, stage, *, chebyshev=False
):
    if samples_per_mode is None:
        return
    num_samples = np.asarray(num_samples, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    mode_index = kappa if chebyshev else kappa / np.pi
    unresolved = mode_index > (num_samples / samples_per_mode)
    if np.any(unresolved):
        axes = np.where(unresolved)[0].tolist()
        warnings.warn(
            "Estimated correction frequency may be under-resolved for "
            f"stage {stage + 1}: axes={axes}, modes={mode_index[unresolved]}, "
            f"samples={num_samples[unresolved]}. Increase "
            "num_samples_for_epsilon/collocation points or lower the frequency.",
            UserWarning,
        )


def _warn_if_training_samples_underresolved(
    kappa, x, samples_per_mode, stage, *, chebyshev=False
):
    if samples_per_mode is None:
        return
    num_samples = min(xi.shape[0] for xi in x)
    num_samples = (num_samples,) * len(kappa)
    _warn_if_frequency_underresolved(
        kappa, num_samples, samples_per_mode, stage, chebyshev=chebyshev
    )


def _residual_below_tolerance(eps_residual, residual_tol):
    if residual_tol is None:
        return False
    return float(np.asarray(eps_residual)) <= residual_tol


def _is_auto_gamma(gamma):
    return isinstance(gamma, str) and gamma == "auto"


def _loss_reduction_rate(initial, final, steps, eps=1e-30):
    initial = float(np.asarray(initial))
    final = float(np.asarray(final))
    if not np.isfinite(initial) or not np.isfinite(final):
        return 0.0
    initial = max(initial, eps)
    final = max(final, eps)
    return max((np.log(initial) - np.log(final)) / max(steps, 1), 0.0)


def _component_loss_pair(component_fun, net, args):
    components = component_fun(net, *args)
    data_loss, equation_loss, _ = _split_pde_loss_components(components)
    return data_loss, equation_loss


def _gamma_trial_rates(
    net,
    component_fun,
    x,
    training_samples,
    optimizer,
    learning_rate,
    steps,
    gamma,
    gamma_g,
    gamma_g_eps,
):
    """Return data/equation convergence rates from a reset pretraining trial."""
    trainable, frozen, static = partition(net)
    gradient_transform = optimizer(learning_rate)
    opt_state = gradient_transform.init(trainable)
    is_lbfgs = optimizer == optax.lbfgs
    x_col = [None] * net.in_size
    args = (*x, training_samples, *x_col)

    def raw_components(trainable):
        trial_net = eqx.combine(trainable, frozen, static)
        return _component_loss_pair(component_fun, trial_net, args)

    def loss(trainable):
        trial_net = eqx.combine(trainable, frozen, static)
        return weighted_pde_loss(
            component_fun(trial_net, *args),
            gamma=gamma,
            gamma_g=gamma_g,
            gamma_g_eps=gamma_g_eps,
        )

    @jit
    def make_step(trainable, opt_state):
        loss_value, grads = value_and_grad(loss)(trainable)
        if is_lbfgs:

            def loss_lbfgs(trainable):
                return loss(trainable)

            updates, opt_state = gradient_transform.update(
                grads,
                opt_state,
                trainable,
                value=loss_value,
                grad=grads,
                value_fn=loss_lbfgs,
            )
        else:
            updates, opt_state = gradient_transform.update(grads, opt_state, trainable)
        trainable = eqx.apply_updates(trainable, updates)
        return trainable, opt_state

    initial_data, initial_equation = raw_components(trainable)
    for _ in range(steps):
        trainable, opt_state = make_step(trainable, opt_state)
    final_data, final_equation = raw_components(trainable)

    return (
        _loss_reduction_rate(initial_data, final_data, steps),
        _loss_reduction_rate(initial_equation, final_equation, steps),
    )


def select_gamma(
    net,
    component_fun,
    x,
    training_samples,
    optimizer,
    learning_rate,
    *,
    initial_gamma=0.5,
    gamma_g=None,
    gamma_g_eps=1e-12,
    steps=100,
    max_trials=5,
    rate_tolerance=0.25,
    bounds=(1e-4, 1.0 - 1e-4),
    adjustment=2.0,
):
    """Estimate a PDE loss weight by balancing component convergence rates.

    This follows the paper's Algorithm 3 structure: for each trial gamma, reset
    to the same input model, pretrain briefly, compare data/equation loss
    convergence rates, adjust gamma, and return the best fixed gamma for the
    real training run.
    """
    if component_fun is None:
        raise ValueError("Automatic gamma selection requires a component loss fun.")
    if steps <= 0 or max_trials <= 0:
        return float(initial_gamma)

    lower, upper = bounds
    gamma = float(np.clip(initial_gamma, lower, upper))
    lower_ratio = max(1.0 - rate_tolerance, 1e-12)
    upper_ratio = 1.0 + rate_tolerance
    best_gamma = gamma
    best_score = np.inf

    for _ in range(max_trials):
        data_rate, equation_rate = _gamma_trial_rates(
            net,
            component_fun,
            x,
            training_samples,
            optimizer,
            learning_rate,
            steps,
            gamma,
            gamma_g,
            gamma_g_eps,
        )
        if equation_rate <= 0 and data_rate <= 0:
            ratio = 1.0
        elif equation_rate <= 0:
            ratio = np.inf
        else:
            ratio = data_rate / equation_rate

        score = abs(np.log(max(ratio, 1e-12))) if np.isfinite(ratio) else np.inf
        if score < best_score:
            best_score = score
            best_gamma = gamma
        if lower_ratio <= ratio <= upper_ratio:
            break
        if ratio < lower_ratio:
            gamma = max(lower, gamma / adjustment)
        else:
            gamma = min(upper, 1.0 - (1.0 - gamma) / adjustment)

    return float(best_gamma)


def _resolve_gamma(
    net,
    component_fun,
    x,
    training_samples,
    optimizer,
    learning_rate,
    gamma,
    gamma_g,
    gamma_g_eps,
    gamma_select_kwargs,
):
    if not _is_auto_gamma(gamma):
        return gamma
    selected = select_gamma(
        net,
        component_fun,
        x,
        training_samples,
        optimizer,
        learning_rate,
        gamma_g=gamma_g,
        gamma_g_eps=gamma_g_eps,
        **gamma_select_kwargs,
    )
    print(f"Selected gamma={selected:.6e} by component pretraining.")
    return selected


def _resolve_loss_fun(
    net,
    loss_fun,
    component_fun,
    x,
    training_samples,
    optimizer,
    learning_rate,
    gamma,
    gamma_g,
    gamma_g_eps,
    gamma_select_kwargs,
):
    if component_fun is None:
        if _is_auto_gamma(gamma):
            raise ValueError("Automatic gamma selection requires a component loss fun.")
        return loss_fun, gamma
    gamma = _resolve_gamma(
        net,
        component_fun,
        x,
        training_samples,
        optimizer,
        learning_rate,
        gamma,
        gamma_g,
        gamma_g_eps,
        gamma_select_kwargs,
    )
    return (
        make_weighted_pde_loss(
            component_fun,
            gamma=gamma,
            gamma_g=gamma_g,
            gamma_g_eps=gamma_g_eps,
        ),
        gamma,
    )


def _resolve_unreduced_loss_fun(
    loss_fun_unreduced,
    residual_component_fun,
    gamma,
    gamma_g,
    gamma_g_eps,
):
    if residual_component_fun is None:
        return loss_fun_unreduced
    return make_weighted_pde_residual_loss(
        residual_component_fun,
        gamma=gamma,
        gamma_g=gamma_g,
        gamma_g_eps=gamma_g_eps,
    )


def _as_correction_specs(extra_stage_corrections, net, stage, stats_values):
    if extra_stage_corrections is None:
        return ()
    if callable(extra_stage_corrections):
        extra_stage_corrections = extra_stage_corrections(net, stage, *stats_values)
    if isinstance(extra_stage_corrections, dict):
        return (extra_stage_corrections,)
    return tuple(extra_stage_corrections)


def _resolve_correction_spec(spec, net, stage, stats_values):
    eps_residual, eps_prediction, kappa = stats_values
    if callable(spec):
        spec = spec(net, stage, eps_residual, eps_prediction, kappa)
    if isinstance(spec, dict):
        epsilon = spec.get("epsilon", spec.get("epsilon_scale", 1.0) * eps_prediction)
        kappa_i = spec.get("kappa", spec.get("kappa_scale", 1.0) * kappa)
        return epsilon, kappa_i
    epsilon, kappa_i = spec
    return epsilon, kappa_i


def _build_next_stage(
    net,
    params,
    key,
    activation,
    eps_prediction,
    kappa,
    width_size,
    depth,
    chebyshev,
    feature_map,
    extra_stage_corrections,
    stage,
    eps_residual,
):
    stats_values = (eps_residual, eps_prediction, kappa)
    epsilons = [eps_prediction]
    kappas = [kappa]
    for spec in _as_correction_specs(extra_stage_corrections, net, stage, stats_values):
        epsilon_i, kappa_i = _resolve_correction_spec(spec, net, stage, stats_values)
        epsilons.append(epsilon_i)
        kappas.append(jnp.asarray(kappa_i))

    common_kwargs = dict(
        width_size=width_size,
        depth=depth,
        params_are_trainable=params is not None,
        chebyshev=chebyshev,
        feature_map=feature_map,
    )
    if len(epsilons) == 1:
        save_kwargs = dict(epsilon=epsilons[0], kappa=kappas[0], **common_kwargs)
        next_net = Stage2(
            net, params=params, key=key, activation=activation, **save_kwargs
        )
        return next_net, save_kwargs

    save_kwargs = dict(
        epsilons=jnp.asarray(epsilons),
        kappas=jnp.stack(kappas),
        **common_kwargs,
    )
    next_net = MultiCorrectionStage(
        net, params=params, key=key, activation=activation, **save_kwargs
    )
    return next_net, save_kwargs


def _restore_train_checkpoint(
    manager,
    step,
    trainable,
    opt_state,
    return_loss_history,
    loss_ref,
):
    """Restore an optimizer checkpoint, allowing older files without loss_ref."""
    last_error = None
    history_sizes = [step + 1, step + 2] if return_loss_history else [None]
    for include_loss_ref in (True, False):
        for history_size in history_sizes:
            restored = {"trainable": trainable, "opt_state": opt_state}
            if include_loss_ref:
                restored["loss_ref"] = loss_ref
            if return_loss_history:
                restored["loss_history"] = jnp.zeros(history_size)
            try:
                restored = manager.restore(
                    step, args=ocp.args.StandardRestore(restored)
                )
            except ValueError as err:
                last_error = err
                continue
            return restored, include_loss_ref
    raise last_error


def _restore_trust_region_checkpoint(manager, step, trainable, loss_ref):
    """Restore a trust-region checkpoint, allowing older files without loss_ref."""
    last_error = None
    for include_loss_ref in (True, False):
        restored = {"trainable": trainable}
        if include_loss_ref:
            restored["loss_ref"] = loss_ref
        try:
            restored = manager.restore(step, args=ocp.args.StandardRestore(restored))
        except ValueError as err:
            last_error = err
            continue
        return restored, include_loss_ref
    raise last_error


def _train(  # noqa: C901
    net,
    loss_fun,
    x,
    training_samples,
    optimizer,
    steps,
    *,
    learning_rate,
    adaptive_sampler=None,
    adaptive_sample_freq=1000,
    adaptive_sample_accumulate=False,
    max_adaptive_samples=None,
    # progress & reproducibility params
    return_loss_history=True,
    print_every=100,
    checkpoint_path=None,
    checkpoint_every=5000,
    resume=False,
    callback=None,
    callback_every=1000,
    key=None,
    debug=False,
    normalize_loss=True,
):
    """Optimize using gradient-based optimization.

    Parameters
    ----------
    net : eqx.Module
        The model to train.
    loss_fun : callable
        Function computing scalar loss with signature ``loss(net,*args)``.
    x : tuple or list of jax.Array
        Input coordinates passed to the loss function.
    training_samples : jax.Array
        Target values for training.
    optimizer : optax.GradientTransformation
        Optax optimizer.
    steps : int
        Total number of training steps.
    learning_rate : float
        Learning rate passed to the optimizer.
    adaptive_sampler : callable, optional
        Function to generate new collocation points.
    adaptive_sample_freq : int, optional
        Frequency of resampling collocation points.
    adaptive_sample_accumulate : bool, optional
        If True, append new adaptive collocation points to the previous set.
        If False, replace the previous adaptive set. Default is False.
    max_adaptive_samples : int, optional
        Maximum number of accumulated adaptive samples per coordinate.
    return_loss_history : bool, optional
        If True, returns the loss history alongside the model.
    print_every : int, optional
        Logging frequency.
    checkpoint_path : str, optional
        Path for saving/restoring checkpoints.
    checkpoint_every : int, optional
        Frequency of checkpoints.
    resume : bool, optional
        If True, resume from an existing checkpoint at ``checkpoint_path``.
        Default is False to avoid silently reusing stale experiments.
    callback : callable, optional
        Function called every ``callback_every`` steps
        with signature ``callback(net,step)``.
    callback_every : int, optional
        Frequency of callback execution.
    key : jax.random.PRNGKey, optional
        Random key for sampling.
    debug : bool, optional
        Whether to print additional details for testing and debugging.
    normalize_loss : bool, optional
        If True, divide this stage's objective by its initial value. This keeps
        later small-amplitude correction stages on a comparable optimizer scale.

    Returns
    -------
    net : eqx.Module
        The trained model.
    loss_history : list of float, optional
        List of loss values per step
        (returned only if ``return_loss_history`` is True).

    """
    is_lbfgs = optimizer == optax.lbfgs
    trainable, frozen, static = partition(net)
    optimizer = optimizer(learning_rate)
    opt_state = optimizer.init(trainable)

    loss_history = []
    print_window = max(print_every, 1)
    last_k_loss = np.zeros(print_window)
    loss_ref = jnp.asarray(1.0)
    loss_ref_restored = False

    manager = None
    start_step = 0
    if checkpoint_path and checkpoint_every > 0:
        manager = checkpoint_manager(checkpoint_path)

        latest_step = manager.latest_step()
        if latest_step is not None and resume:
            start_step = latest_step
            print(f"\n=== Resuming training from step {start_step} ===")
            restored, loss_ref_restored = _restore_train_checkpoint(
                manager,
                start_step,
                trainable,
                opt_state,
                return_loss_history,
                loss_ref,
            )

            trainable = restored["trainable"]
            opt_state = restored["opt_state"]
            loss_ref = restored.get("loss_ref", loss_ref)
            net = eqx.combine(trainable, frozen, static)

            if return_loss_history:
                loss_history = np.array(restored["loss_history"])
                last_k_loss = _fill_lask_k_buffer(last_k_loss, loss_history)
                loss_history = list(loss_history)

            start_step += 1
        elif latest_step is not None:
            warnings.warn(
                f"Existing checkpoint at {checkpoint_path!r} was not restored "
                "because resume=False.",
                UserWarning,
            )

    if debug:
        print("\n-----   Static  -----")
        print(static)
        print("\n----- Trainable -----")
        print(trainable)
        print("\n-----   Frozen  -----")
        print(frozen)
        print("\n----- Recombine -----")
        print(eqx.combine(trainable, frozen, static))

    def raw_loss(trainable, frozen, static, *args):
        net = eqx.combine(trainable, frozen, static)
        return loss_fun(net, *args)

    def loss(trainable, frozen, static, loss_ref, *args):
        return raw_loss(trainable, frozen, static, *args) / loss_ref

    @partial(jit, static_argnames=["static"])
    def make_step(trainable, frozen, static, opt_state, loss_ref, *args):
        loss_value, grads = value_and_grad(loss)(
            trainable, frozen, static, loss_ref, *args
        )
        if is_lbfgs:

            def loss_lbfgs(trainable):
                return loss(trainable, frozen, static, loss_ref, *args)

            updates, opt_state = optimizer.update(
                grads,
                opt_state,
                trainable,
                value=loss_value,
                grad=grads,
                value_fn=loss_lbfgs,
            )
        else:
            updates, opt_state = optimizer.update(grads, opt_state, trainable)
        trainable = eqx.apply_updates(trainable, updates)
        return trainable, opt_state, loss_value

    print(f"--- Values at step {start_step} ---")
    net.print_params()
    if debug:
        net.print_frozen_params()

    print("--- Starting training ---")
    start_train_time = time.time()

    x_col = [None] * net.in_size
    if (
        (adaptive_sampler is not None)
        and (adaptive_sample_freq > 0)
        and (start_step < steps)
    ):
        print(f"Resampled at step {start_step}.")
        x_new, key = adaptive_sampler(eqx.combine(trainable, frozen, static), key=key)
        x_col = _merge_adaptive_samples(
            x_col,
            x_new,
            accumulate=adaptive_sample_accumulate,
            max_samples=max_adaptive_samples,
        )

    if not normalize_loss:
        loss_ref = jnp.asarray(1.0)
    elif loss_ref_restored:
        print(f"Restored loss normalization scale: {loss_ref:.6e}.")
    elif start_step < steps:
        loss_ref = _safe_loss_ref(
            raw_loss(trainable, frozen, static, *x, training_samples, *x_col)
        )
        print(f"Initial loss normalization scale: {loss_ref:.6e}.")

    for step in range(start_step, steps):
        if (
            (adaptive_sampler is not None)
            and (adaptive_sample_freq > 0)
            and (step > start_step)
            and (step % adaptive_sample_freq == 0)
        ):
            print(f"Resampled at step {step}.")
            x_new, key = adaptive_sampler(
                eqx.combine(trainable, frozen, static), key=key
            )
            x_col = _merge_adaptive_samples(
                x_col,
                x_new,
                accumulate=adaptive_sample_accumulate,
                max_samples=max_adaptive_samples,
            )

        trainable, opt_state, loss_value = make_step(
            trainable,
            frozen,
            static,
            opt_state,
            loss_ref,
            *x,
            training_samples,
            *x_col,
        )

        if return_loss_history:
            loss_history.append(loss_value)

        last_k_loss[step % print_window] = loss_value
        if _is_multiple_or_last(step, print_every, steps):
            last_k = loss_value if (step == 0) else last_k_loss.mean()
            loss_window_name = print_every if print_every > 0 else print_window
            print(
                f"Step {step}, Loss: {loss_value:.6e}, "
                f"Last {loss_window_name} mean loss: {last_k}.",
                flush=True,
            )
            net = eqx.combine(trainable, frozen, static)
            net.print_params()
            if debug:
                net.print_frozen_params()

        if (
            manager
            and (checkpoint_every > 0)
            and _is_completed_multiple_or_last(step, checkpoint_every, steps)
        ):
            man_args = {
                "trainable": trainable,
                "opt_state": opt_state,
                "loss_ref": loss_ref,
            }
            if len(loss_history):
                man_args["loss_history"] = jnp.asarray(loss_history)
            manager.save(step, args=ocp.args.StandardSave(man_args))

        if callback and _is_multiple_or_last(step, callback_every, steps):
            current_net = eqx.combine(trainable, frozen, static)
            callback(current_net, step)

    end_train_time = time.time()
    print("--- Finished training ---\n")
    print(f"Training time: {end_train_time - start_train_time:.2f}")

    if manager:
        manager.wait_until_finished()
        manager.close()

    net = eqx.combine(trainable, frozen, static)
    return (net, loss_history) if return_loss_history else net


def _trust_region_train(  # noqa: C901
    net,
    loss_fun_unreduced,
    x,
    training_samples,
    steps,
    *,
    rtol,
    atol,
    linear_solver,
    adaptive_sampler=None,
    adaptive_sample_freq=100,
    adaptive_sample_accumulate=False,
    max_adaptive_samples=None,
    # progress & reproducibility params
    checkpoint_path=None,
    checkpoint_every=100,
    resume=False,
    callback=None,
    callback_every=100,
    key=None,
    normalize_loss=True,
):
    """Optimize using a Levenberg-Marquardt nonlinear least squares solver.

    Parameters
    ----------
    net : eqx.Module
        The model to train.
    loss_fun_unreduced : callable
        Function returning a residual vector for least squares.
    x : tuple or list of jax.Array
        Input coordinates passed to the loss function.
    training_samples : jax.Array
        Target values.
    steps : int
        Maximum number of solver steps.
    rtol : float
        Relative tolerance for the solver.
    atol : float
        Absolute tolerance for the solver.
    linear_solver : lineax.AbstractLinearSolver
        The linear solver used to compute the Gauss-Newton step.
    adaptive_sampler : callable, optional
        Function to generate new collocation points.
    adaptive_sample_freq : int, optional
        Frequency of resampling collocation points.
    adaptive_sample_accumulate : bool, optional
        If True, append new adaptive collocation points to the previous set.
        If False, replace the previous adaptive set. Default is False.
    max_adaptive_samples : int, optional
        Maximum number of accumulated adaptive samples per coordinate.
    checkpoint_path : str, optional
        Path for saving/restoring checkpoints.
    checkpoint_every : int, optional
        Frequency of checkpoints.
    resume : bool, optional
        If True, resume from an existing checkpoint at ``checkpoint_path``.
        Default is False to avoid silently reusing stale experiments.
    callback : callable, optional
        Function called every ``callback_every`` steps
        with signature ``callback(net,step)``.
    callback_every : int, optional
        Frequency of callback execution.
    key : jax.random.PRNGKey, optional
        Random key for sampling.
    normalize_loss : bool, optional
        If True, scale the residual vector by the square root of its initial
        least-squares objective for this stage.

    Returns
    -------
    net : eqx.Module
        The trained model.

    """
    trainable, frozen, static = partition(net)
    loss_ref = jnp.asarray(1.0)
    loss_ref_restored = False
    solver = optimistix.LevenbergMarquardt(
        rtol=rtol,
        atol=atol,
        verbose=False,
        linear_solver=linear_solver,
    )

    manager = None
    start_step = 0
    if checkpoint_path and checkpoint_every > 0:
        manager = checkpoint_manager(checkpoint_path)

        latest_step = manager.latest_step()
        if latest_step is not None and resume:
            start_step = latest_step
            print(f"\n=== Resuming training from step {start_step} ===")
            restored, loss_ref_restored = _restore_trust_region_checkpoint(
                manager, start_step, trainable, loss_ref
            )
            trainable = restored["trainable"]
            loss_ref = restored.get("loss_ref", loss_ref)
            net = eqx.combine(trainable, frozen, static)
        elif latest_step is not None:
            warnings.warn(
                f"Existing checkpoint at {checkpoint_path!r} was not restored "
                "because resume=False.",
                UserWarning,
            )

    def raw_loss(trainable, args):
        frozen, static, *rest = args
        net = eqx.combine(trainable, frozen, static)
        return loss_fun_unreduced(net, *rest)

    def loss(trainable, args):
        loss_ref, frozen, static, *rest = args
        raw_residual = raw_loss(trainable, (frozen, static, *rest))
        return raw_residual / jnp.sqrt(loss_ref)

    print(f"--- Values at step {start_step} ---")
    net.print_params()

    print("--- Starting training ---")
    start_train_time = time.time()

    x_col = [None] * net.in_size
    if start_step >= steps:
        end_train_time = time.time()
        print("--- Finished training ---\n")
        print(f"Training time: {end_train_time - start_train_time:.2f}")
        if manager:
            manager.wait_until_finished()
            manager.close()
        return net

    step_frequencies = [steps]
    if adaptive_sampler is not None and adaptive_sample_freq > 0:
        step_frequencies.append(adaptive_sample_freq)
    if checkpoint_every > 0:
        step_frequencies.append(checkpoint_every)
    step_per_opt = min(step_frequencies)
    for step in range(start_step, steps, step_per_opt):
        steps_this_chunk = min(step_per_opt, steps - step)
        if (
            (adaptive_sampler is not None)
            and (adaptive_sample_freq > 0)
            and (
                step == start_step
                or step // adaptive_sample_freq
                > (step - step_per_opt) // adaptive_sample_freq
            )
        ):
            print(f"Resampled at step {step}.")
            x_new, key = adaptive_sampler(
                eqx.combine(trainable, frozen, static), key=key
            )
            x_col = _merge_adaptive_samples(
                x_col,
                x_new,
                accumulate=adaptive_sample_accumulate,
                max_samples=max_adaptive_samples,
            )

        if not normalize_loss:
            loss_ref = jnp.asarray(1.0)
        elif loss_ref_restored and (step == start_step):
            print(f"Restored loss normalization scale: {loss_ref:.6e}.")
        elif step == start_step:
            initial_residual = raw_loss(
                trainable, (frozen, static, *x, training_samples, *x_col)
            )
            loss_ref = _safe_loss_ref(jnp.sum(initial_residual**2))
            print(f"Initial loss normalization scale: {loss_ref:.6e}.")

        sol = optimistix.least_squares(
            loss,
            solver,
            trainable,
            args=(loss_ref, frozen, static, *x, training_samples, *x_col),
            max_steps=steps_this_chunk,
            throw=False,
        )
        trainable = sol.value
        print(optimistix.RESULTS[sol.result])
        # can't extract loss history programmatically from optimistix solver:
        #    https://github.com/patrick-kidger/optimistix/issues/52

        net = eqx.combine(trainable, frozen, static)
        net.print_params()

        current_step = step + steps_this_chunk
        if (
            manager
            and (checkpoint_every > 0)
            and (
                (current_step // checkpoint_every > step // checkpoint_every)
                or (current_step == steps)
            )
        ):
            manager.save(
                current_step,
                args=ocp.args.StandardSave(
                    {"trainable": trainable, "loss_ref": loss_ref}
                ),
            )

        if (
            callback
            and (callback_every > 0)
            and (
                (current_step // callback_every > step // callback_every)
                or (current_step == steps)
            )
        ):
            current_net = eqx.combine(trainable, frozen, static)
            callback(current_net, current_step)

    end_train_time = time.time()
    print("--- Finished training ---\n")
    print(f"Training time: {end_train_time - start_train_time:.2f}")

    if manager:
        manager.wait_until_finished()
        manager.close()

    net = eqx.combine(trainable, frozen, static)
    return net


def multistage_train(
    net,
    residual_fun_s1,
    residual_fun_s2,
    loss_fun_s1,
    loss_fun_s2,
    x,
    training_samples,
    optimizer,
    steps,
    *,
    learning_rate=None,
    adaptive_sample_freq=1000,
    n_stages=2,
    width_size=20,
    depth=4,
    activation=jnp.tanh,
    num_samples_for_epsilon=(1024,),
    order=(1,),
    beta_fun=None,
    heuristic=0.9,
    frequency_estimator="spectral",
    residual_tol=0.0,
    frequency_samples_per_mode=6.0,
    chebyshev=False,
    feature_map="separable",
    stage_correction_param_map=None,
    extra_stage_corrections=None,
    x_stage2=None,
    training_samples_stage2=None,
    loss_components_fun_s1=None,
    loss_components_fun_s2=None,
    gamma_s1=0.5,
    gamma_s2=0.5,
    gamma_g_s1=None,
    gamma_g_s2=None,
    gamma_g_eps=1e-12,
    gamma_select_kwargs=None,
    adaptive_sample_accumulate=False,
    max_adaptive_samples=None,
    # progress & reproducibility params
    return_loss_history=True,
    print_every=100,
    key=None,
    net_kwargs_for_save=None,
    name="",
    checkpoint_dir=None,
    checkpoint_every=5000,
    resume=False,
    benchmark_state=None,
    normalize_loss=True,
    **adaptive_sample_kwargs,
):
    """Multi-stage training.

    Examples
    --------
      * See ``tests/test_burgers.py``.

    Parameters
    ----------
    net : eqx.Module
        The initial model architecture.
    residual_fun_s1 : callable
        Function to compute PDE residuals for the first stage.
    residual_fun_s2 : callable
        Function to compute PDE residuals for subsequent stages.
    loss_fun_s1 : callable
        Scalar output loss function for the first stage.
    loss_fun_s2 : callable
        Scalar output loss function for subsequent stages.
    x : tuple or list of jax.Array
        Input coordinates for the first stage.
    training_samples : jax.Array
        Target values for the first stage.
    optimizer : optax.GradientTransformation
        Optimizer for training loops.
    steps : int
        Number of training steps per stage.
    learning_rate : float, optional
        Learning rate passed to the optimizer.
    adaptive_sample_freq : int, optional
        Frequency of adaptive sampling during training.
    n_stages : int, optional
        Total number of training stages. Default is 2.
    width_size : int, optional
        Width of the sub-networks added in later stages.
    depth : int, optional
        Depth of the sub-networks added in later stages.
    activation : callable, optional
        Activation function for new stages. Default is ``jnp.tanh``.
    num_samples_for_epsilon : tuple, optional
        Number of samples used to estimate error statistics between stages.
    order : tuple, optional
        Derivative orders used for error estimation. A 1D tuple is interpreted
        as separate operator terms in each input direction; pass nested
        multi-indices for mixed derivative terms.
    beta_fun : callable, optional
        Function defining scalar or per-operator-term coefficients for the
        error estimate.
    heuristic : float, optional
        Heuristic multiplier for error estimation. Default is 0.9.
        Used only when ``frequency_estimator="zero_crossing"``.
    frequency_estimator : {"spectral", "zero_crossing"}
        Fourier residual frequency estimator used between stages.
    residual_tol : float or None
        Stop adding stages when the RMS residual estimate is at or below this
        tolerance. Set to None to disable this safeguard.
    frequency_samples_per_mode : float or None
        Warn when the estimated Fourier/Chebyshev mode exceeds the available
        statistics samples divided by this value. Set to None to disable.
    chebyshev : bool
        Whether to use Chebyshev feature mapping instead of Fourier.
        If given, ``heuristic`` is ignored.
    feature_map : {"separable", "random"}
        First-layer feature geometry for new stages. ``"separable"`` is the
        default; ``"random"`` preserves the previous dense plane-wave mapping.
    stage_correction_param_map : dict, optional
        Mapping from current-stage parameter names to next-stage correction
        names or initializers. Custom entries extend the built-in defaults,
        including ``{"log_lambda_2": "lambda_2"}`` for Burgers-style signed
        physical diffusion corrections.
    extra_stage_corrections : sequence or callable, optional
        Additional correction specs for automatically constructing a
        ``MultiCorrectionStage``. Each spec may be ``(epsilon, kappa)``, a dict
        with ``epsilon``/``kappa`` or ``epsilon_scale``/``kappa_scale``, or a
        callable ``spec(net, stage, eps_residual, eps_prediction, kappa)``.
    x_stage2 : tuple of jax.Array, optional
        Input coordinates for stage 2 and beyond. Default is ``x``.
    training_samples_stage2 : jax.Array, optional
        Training data for stage 2 and beyond. Default is ``training_samples``.
    loss_components_fun_s1, loss_components_fun_s2 : callable, optional
        Component loss functions returning data/equation/(optional) gradient
        losses. When supplied, they are wrapped using ``gamma`` and ``gamma_g``.
    gamma_s1, gamma_s2 : float, optional
        Equation-loss weights used with component loss functions.
    gamma_g_s1, gamma_g_s2 : float, optional
        Residual-gradient weights used with component loss functions. If None
        and a gradient component is returned, the weight is estimated from
        residual and residual-gradient magnitudes.
    gamma_g_eps : float, optional
        Numerical floor for automatic ``gamma_g`` estimates.
    gamma_select_kwargs : dict, optional
        Options for automatic Algorithm-3-style gamma selection. Set
        ``gamma_s1="auto"`` and/or ``gamma_s2="auto"`` to enable it.
    adaptive_sample_accumulate : bool, optional
        If True, adaptive collocation points accumulate instead of replacing
        the previous adaptive set.
    max_adaptive_samples : int, optional
        Maximum number of accumulated adaptive samples per coordinate.
    return_loss_history : bool, optional
        If True, returns loss histories for all stages.
    print_every : int, optional
        Logging frequency.
    key : jax.random.PRNGKey, optional
        Random key for initialization and sampling.
    net_kwargs_for_save : dict, optional
        Additional metadata to save with the model.
    name : str, optional
        Base name for saving models and checkpoints.
    checkpoint_dir : str, optional
        Directory to store stage-specific checkpoints. Default is None, which
        disables checkpointing unless explicitly requested.
    checkpoint_every : int, optional
        Frequency of checkpointing within stages.
    resume : bool, optional
        If True, resume each stage from an existing checkpoint. Default is
        False to avoid silently reusing stale experiments.
    benchmark_state : callable, optional
        Callback for external benchmarking or logging. Signature:
        ``benchmark_state(net,stage,name,step=step)``.
    normalize_loss : bool, optional
        Whether to normalize each stage's objective by its initial value.

    Returns
    -------
    net : eqx.Module
        The final trained multi-stage model.
    loss_histories : list of list of float, optional
        A list containing the loss history for each stage
        (if `return_loss_history`` is True).

    """
    if key is None:
        key = jax.random.PRNGKey(42)
    if net_kwargs_for_save is None:
        net_kwargs_for_save = {}
    if x_stage2 is None:
        x_stage2 = x
    if training_samples_stage2 is None:
        training_samples_stage2 = training_samples
    if gamma_select_kwargs is None:
        gamma_select_kwargs = {}

    adaptive_sample_kwargs_base = dict(adaptive_sample_kwargs)

    residual_fun = residual_fun_s1
    loss_fun_base = loss_fun_s1
    loss_components_fun = loss_components_fun_s1
    gamma = gamma_s1
    gamma_g = gamma_g_s1
    loss_histories = []

    for stage in range(n_stages):
        loss_fun, _ = _resolve_loss_fun(
            net,
            loss_fun_base,
            loss_components_fun,
            x,
            training_samples,
            optimizer,
            learning_rate,
            gamma,
            gamma_g,
            gamma_g_eps,
            gamma_select_kwargs,
        )

        current_callback = None
        if benchmark_state is not None:

            def current_callback(n, s):  # noqa: F811
                benchmark_state(n, stage, name, step=s)

        if adaptive_sample_freq > 0:
            stage_adaptive_sample_kwargs = dict(adaptive_sample_kwargs_base)
            stage_adaptive_sample_kwargs.setdefault("n_candidates", len(x[0]) * 10)
            stage_adaptive_sample_kwargs.setdefault("n_selected", len(x[0]) // 2)
            adaptive_sampler = partial(
                adaptive_sample,
                residual_fun=residual_fun,
                in_size=net.in_size,
                **stage_adaptive_sample_kwargs,
            )
        else:
            adaptive_sampler = None

        key, train_key = jax.random.split(key)
        net = _train(
            net=net,
            loss_fun=loss_fun,
            x=x,
            training_samples=training_samples,
            optimizer=optimizer,
            steps=steps,
            learning_rate=learning_rate,
            adaptive_sampler=adaptive_sampler,
            adaptive_sample_freq=adaptive_sample_freq,
            adaptive_sample_accumulate=adaptive_sample_accumulate,
            max_adaptive_samples=max_adaptive_samples,
            return_loss_history=return_loss_history,
            print_every=print_every,
            checkpoint_path=_checkpoint_stage_path(
                checkpoint_dir, f"{name}_stage_{stage}"
            ),
            checkpoint_every=checkpoint_every,
            resume=resume,
            callback=current_callback,
            key=train_key,
            normalize_loss=normalize_loss,
        )
        if return_loss_history:
            net, loss_history = net
            loss_histories.append(loss_history)

        save(f"models/{name}_net_stage_{stage}.eqx", net, **net_kwargs_for_save)

        if benchmark_state is not None:
            benchmark_state(net, stage, name)

        if stage == (n_stages - 1):
            continue

        non_static, static = eqx.partition(net, eqx.is_inexact_array)
        eps_residual, eps_prediction, kappa = (stats_chebyshev if chebyshev else stats)(
            non_static,
            static,
            residual_fun,
            num_samples_for_epsilon,
            order,
            beta_fun,
            **(
                {}
                if chebyshev
                else {
                    "heuristic": heuristic,
                    "frequency_estimator": frequency_estimator,
                }
            ),
        )
        print(f"Stage {stage} statistics:")
        print(f"RMS residual estimate used for stage {stage +1} is {eps_residual}.")
        print(f"RMS prediction residual used for stage {stage +1} is {eps_prediction}.")
        print(f"Estimate frequency kappa used for stage {stage +1} is {kappa}.")

        if _residual_below_tolerance(eps_residual, residual_tol):
            print(
                f"Stopping after stage {stage}: RMS residual {eps_residual} "
                f"is at or below residual_tol={residual_tol}."
            )
            break

        stats_num_samples = _expanded_num_samples(num_samples_for_epsilon, net.in_size)
        _warn_if_frequency_underresolved(
            kappa,
            stats_num_samples,
            frequency_samples_per_mode,
            stage,
            chebyshev=chebyshev,
        )
        next_stage_x = x_stage2 if stage == 0 else x
        _warn_if_training_samples_underresolved(
            kappa,
            next_stage_x,
            frequency_samples_per_mode,
            stage,
            chebyshev=chebyshev,
        )

        params = _stage_correction_params_or_none(
            getattr(net, "_params", None), stage_correction_param_map
        )

        key, subkey = jax.random.split(key)
        net, net_kwargs_for_save = _build_next_stage(
            net,
            params,
            subkey,
            activation,
            eps_prediction,
            kappa,
            width_size=width_size,
            depth=depth,
            chebyshev=chebyshev,
            feature_map=feature_map,
            extra_stage_corrections=extra_stage_corrections,
            stage=stage,
            eps_residual=eps_residual,
        )

        residual_fun = residual_fun_s2
        loss_fun_base = loss_fun_s2
        loss_components_fun = loss_components_fun_s2
        gamma = gamma_s2
        gamma_g = gamma_g_s2
        # Next stages will use same as stage 2 data currently.
        training_samples = training_samples_stage2
        x = x_stage2

    return (net, loss_histories) if return_loss_history else net


def multistage_trust_region_train(
    net,
    residual_fun_s1,
    residual_fun_s2,
    loss_fun_s1,
    loss_fun_s2,
    loss_fun_s1_unreduced,
    loss_fun_s2_unreduced,
    x,
    training_samples,
    steps,
    lbfgs_steps=1000,
    *,
    rtol=1e-7,
    atol=1e-8,
    rtol_decay_factor=0.7,
    atol_decay_factor=0.1,
    linear_solver=(lx.QR(), lx.Normal(lx.CG(rtol=1e-7, atol=1e-7))),
    learning_rate=None,
    adaptive_sample_freq=100,
    n_stages=2,
    width_size=20,
    depth=4,
    activation=jnp.tanh,
    num_samples_for_epsilon=(1024,),
    order=(1,),
    beta_fun=None,
    heuristic=0.9,
    frequency_estimator="spectral",
    residual_tol=0.0,
    frequency_samples_per_mode=6.0,
    chebyshev=False,
    feature_map="separable",
    stage_correction_param_map=None,
    extra_stage_corrections=None,
    x_stage2=None,
    training_samples_stage2=None,
    loss_components_fun_s1=None,
    loss_components_fun_s2=None,
    loss_residual_components_fun_s1=None,
    loss_residual_components_fun_s2=None,
    gamma_s1=0.5,
    gamma_s2=0.5,
    gamma_g_s1=None,
    gamma_g_s2=None,
    gamma_g_eps=1e-12,
    gamma_select_kwargs=None,
    adaptive_sample_accumulate=False,
    max_adaptive_samples=None,
    # Progress & reproducibility params
    print_every=100,
    key=None,
    net_kwargs_for_save=None,
    name="",
    checkpoint_dir=None,
    checkpoint_every=100,
    resume=False,
    benchmark_state=None,
    normalize_loss=True,
    **adaptive_sample_kwargs,
):
    """Multi-stage training using trust region based optimization.

    Examples
    --------
      * See ``tests/test_burgers.py``.

    Parameters
    ----------
    net : eqx.Module
        The initial model architecture.
    residual_fun_s1 : callable
        Function to compute PDE residuals for the first stage.
    residual_fun_s2 : callable
        Function to compute PDE residuals for subsequent stages.
    loss_fun_s1 : callable
        Scalar output loss function for the first stage.
    loss_fun_s2 : callable
        Scalar output loss function for subsequent stages.
    loss_fun_s1_unreduced : callable
        Unreduced loss function for the first stage.
    loss_fun_s2_unreduced : callable
        Unreduced loss function for subsequent stages.
    x : tuple or list of jax.Array
        Input coordinates passed to the residual function for the first stage.
    training_samples : jax.Array
        Target values for the first stage.
    steps : int
        Maximum number of solver steps per stage.
    lbfgs_steps : int
        Number of steps to use LBFGS prior to trust region approach.
        Default is 1000.
    rtol : float
        Relative tolerance for the Levenberg-Marquardt solver for the first stage.
    atol : float
        Absolute tolerance for the Levenberg-Marquardt solver for the first stage.
    rtol_decay_factor : float
        Decay factor for ``rtol`` in subsequent stages.
        Default of ``0.7`` means the next stage will have ``new_rtol=rtol*0.7``.
    atol_decay_factor : float
        Decay factor for ``atol`` in subsequent stages.
        Default of ``0.1`` means the next stage will have ``new_atol=atol*0.1``.
    linear_solver : tuple[lineax.AbstractLinearSolver]
        The linear solver used to compute the Gauss-Newton step.
        Default is QR for first stage and conjugate gradient on the normal
        equations for following stages.
    learning_rate : float, optional
        Learning rate passed to the LBFGS warmup optimizer. The
        Levenberg-Marquardt phase is controlled by ``rtol``, ``atol``, and
        ``linear_solver``.
    adaptive_sample_freq : int, optional
        Frequency of adaptive sampling during training with trust region method.
        Default is 100. LBFGS warmup steps will adaptive sample with
        10 times less frequency.
    n_stages : int, optional
        Total number of training stages. Default is 2.
    width_size : int, optional
        Width of the sub-networks added in later stages.
    depth : int, optional
        Depth of the sub-networks added in later stages.
    activation : callable, optional
        Activation function for new stages. Default is ``jnp.tanh``.
    num_samples_for_epsilon : tuple, optional
        Number of samples used to estimate error statistics between stages.
    order : tuple, optional
        Derivative orders used for error estimation. A 1D tuple is interpreted
        as separate operator terms in each input direction; pass nested
        multi-indices for mixed derivative terms.
    beta_fun : callable, optional
        Function defining scalar or per-operator-term coefficients for the
        error estimate.
    heuristic : float, optional
        Heuristic multiplier for error estimation. Default is 0.9.
        Used only when ``frequency_estimator="zero_crossing"``.
    frequency_estimator : {"spectral", "zero_crossing"}
        Fourier residual frequency estimator used between stages.
    residual_tol : float or None
        Stop adding stages when the RMS residual estimate is at or below this
        tolerance. Set to None to disable this safeguard.
    frequency_samples_per_mode : float or None
        Warn when the estimated Fourier/Chebyshev mode exceeds the available
        statistics samples divided by this value. Set to None to disable.
    chebyshev : bool
        Whether to use Chebyshev feature mapping instead of Fourier.
        If given, ``heuristic`` is ignored.
    feature_map : {"separable", "random"}
        First-layer feature geometry for new stages. ``"separable"`` is the
        default; ``"random"`` preserves the previous dense plane-wave mapping.
    stage_correction_param_map : dict, optional
        Mapping from current-stage parameter names to next-stage correction
        names or initializers. Custom entries extend the built-in defaults,
        including ``{"log_lambda_2": "lambda_2"}`` for Burgers-style signed
        physical diffusion corrections.
    extra_stage_corrections : sequence or callable, optional
        Additional correction specs for automatically constructing a
        ``MultiCorrectionStage``. Each spec may be ``(epsilon, kappa)``, a dict
        with ``epsilon``/``kappa`` or ``epsilon_scale``/``kappa_scale``, or a
        callable ``spec(net, stage, eps_residual, eps_prediction, kappa)``.
    x_stage2 : tuple of jax.Array, optional
        Input coordinates for stage 2 and beyond. Default is ``x``.
    training_samples_stage2 : jax.Array, optional
        Training data for stage 2 and beyond. Default is ``training_samples``.
    loss_components_fun_s1, loss_components_fun_s2 : callable, optional
        Component loss functions returning data/equation/(optional) gradient
        losses. When supplied, they are wrapped using ``gamma`` and ``gamma_g``.
    loss_residual_components_fun_s1, loss_residual_components_fun_s2 : callable
        Unreduced component residual functions returning
        data/equation/(optional) gradient residual vectors. When supplied, they
        are weighted consistently with the scalar component objective for the
        Levenberg-Marquardt phase.
    gamma_s1, gamma_s2 : float, optional
        Equation-loss weights used with component loss functions.
    gamma_g_s1, gamma_g_s2 : float, optional
        Residual-gradient weights used with component loss functions. If None
        and a gradient component is returned, the weight is estimated from
        residual and residual-gradient magnitudes.
    gamma_g_eps : float, optional
        Numerical floor for automatic ``gamma_g`` estimates.
    gamma_select_kwargs : dict, optional
        Options for automatic Algorithm-3-style gamma selection. Set
        ``gamma_s1="auto"`` and/or ``gamma_s2="auto"`` to enable it.
    adaptive_sample_accumulate : bool, optional
        If True, adaptive collocation points accumulate instead of replacing
        the previous adaptive set.
    max_adaptive_samples : int, optional
        Maximum number of accumulated adaptive samples per coordinate.
    print_every : int, optional
        Logging frequency.
    key : jax.random.PRNGKey, optional
        Random key for initialization and sampling.
    net_kwargs_for_save : dict, optional
        Additional metadata to save with the model.
    name : str, optional
        Base name for saving models and checkpoints.
    checkpoint_dir : str, optional
        Directory to store stage-specific checkpoints. Default is None, which
        disables checkpointing unless explicitly requested.
    checkpoint_every : int, optional
        Frequency of checkpointing within stages. Default is 100.
        LBFGS warmup steps will checkpoint with 10 times less frequency.
    resume : bool, optional
        If True, resume each stage from an existing checkpoint. Default is
        False to avoid silently reusing stale experiments.
    benchmark_state : callable, optional
        Callback for external benchmarking or logging. Signature:
        ``benchmark_state(net,stage,name,step=step)``.
    normalize_loss : bool, optional
        Whether to normalize each stage's objective by its initial value.

    Returns
    -------
    net : eqx.Module
        The final trained multi-stage model.

    """
    if key is None:
        key = jax.random.PRNGKey(42)
    if net_kwargs_for_save is None:
        net_kwargs_for_save = {}
    if x_stage2 is None:
        x_stage2 = x
    if training_samples_stage2 is None:
        training_samples_stage2 = training_samples
    if gamma_select_kwargs is None:
        gamma_select_kwargs = {}

    adaptive_sample_kwargs_base = dict(adaptive_sample_kwargs)

    residual_fun = residual_fun_s1
    loss_fun_base = loss_fun_s1
    loss_fun_unreduced_base = loss_fun_s1_unreduced
    loss_components_fun = loss_components_fun_s1
    loss_residual_components_fun = loss_residual_components_fun_s1
    gamma = gamma_s1
    gamma_g = gamma_g_s1
    linear_solver, linear_solver_next = linear_solver

    for stage in range(n_stages):
        loss_fun, selected_gamma = _resolve_loss_fun(
            net,
            loss_fun_base,
            loss_components_fun,
            x,
            training_samples,
            optax.lbfgs,
            learning_rate,
            gamma,
            gamma_g,
            gamma_g_eps,
            gamma_select_kwargs,
        )
        loss_fun_unreduced = _resolve_unreduced_loss_fun(
            loss_fun_unreduced_base,
            loss_residual_components_fun,
            selected_gamma,
            gamma_g,
            gamma_g_eps,
        )
        if loss_components_fun is not None and loss_residual_components_fun is None:
            warnings.warn(
                "Scalar component loss weights do not alter the trust-region "
                "unreduced residual. Pass loss_residual_components_fun_s1/s2 "
                "to apply gamma/gamma_g to the Levenberg-Marquardt phase.",
                UserWarning,
            )

        current_callback = None
        if benchmark_state is not None:

            def current_callback(n, s):  # noqa: F811
                benchmark_state(n, stage, name, step=s)

        if adaptive_sample_freq > 0:
            stage_adaptive_sample_kwargs = dict(adaptive_sample_kwargs_base)
            stage_adaptive_sample_kwargs.setdefault("n_candidates", len(x[0]) * 10)
            stage_adaptive_sample_kwargs.setdefault("n_selected", len(x[0]) // 2)
            adaptive_sampler = partial(
                adaptive_sample,
                residual_fun=residual_fun,
                in_size=net.in_size,
                **stage_adaptive_sample_kwargs,
            )
        else:
            adaptive_sampler = None

        key, train_key = jax.random.split(key)
        net, _ = _train(
            net=net,
            loss_fun=loss_fun,
            x=x,
            training_samples=training_samples,
            optimizer=optax.lbfgs,
            steps=lbfgs_steps,
            learning_rate=learning_rate,
            adaptive_sampler=adaptive_sampler,
            adaptive_sample_freq=adaptive_sample_freq * 10,
            adaptive_sample_accumulate=adaptive_sample_accumulate,
            max_adaptive_samples=max_adaptive_samples,
            return_loss_history=True,
            print_every=print_every,
            checkpoint_path=_checkpoint_stage_path(
                checkpoint_dir, f"{name}_lbfgs_warmup_stage_{stage}"
            ),
            checkpoint_every=checkpoint_every * 10,
            resume=resume,
            callback=current_callback,
            key=train_key,
            normalize_loss=normalize_loss,
        )

        key, train_key = jax.random.split(key)
        net = _trust_region_train(
            net=net,
            loss_fun_unreduced=loss_fun_unreduced,
            x=x,
            training_samples=training_samples,
            steps=steps,
            rtol=rtol,
            atol=atol,
            linear_solver=linear_solver,
            adaptive_sampler=adaptive_sampler,
            adaptive_sample_freq=adaptive_sample_freq,
            adaptive_sample_accumulate=adaptive_sample_accumulate,
            max_adaptive_samples=max_adaptive_samples,
            checkpoint_path=_checkpoint_stage_path(
                checkpoint_dir, f"{name}_stage_{stage}"
            ),
            checkpoint_every=checkpoint_every,
            resume=resume,
            callback=current_callback,
            key=train_key,
            normalize_loss=normalize_loss,
        )

        linear_solver = linear_solver_next
        loss_fun_unreduced_base = loss_fun_s2_unreduced
        rtol *= rtol_decay_factor
        atol *= atol_decay_factor

        save(f"models/{name}_net_stage_{stage}.eqx", net, **net_kwargs_for_save)

        if benchmark_state is not None:
            benchmark_state(net, stage, name)

        if stage == (n_stages - 1):
            continue

        non_static, static = eqx.partition(net, eqx.is_inexact_array)
        eps_residual, eps_prediction, kappa = (stats_chebyshev if chebyshev else stats)(
            non_static,
            static,
            residual_fun,
            num_samples_for_epsilon,
            order,
            beta_fun,
            **(
                {}
                if chebyshev
                else {
                    "heuristic": heuristic,
                    "frequency_estimator": frequency_estimator,
                }
            ),
        )
        print(f"Stage {stage} statistics:")
        print(f"RMS residual estimate used for stage {stage +1} is {eps_residual}.")
        print(f"RMS prediction residual used for stage {stage +1} is {eps_prediction}.")
        print(f"Estimate frequency kappa used for stage {stage +1} is {kappa}.")

        if _residual_below_tolerance(eps_residual, residual_tol):
            print(
                f"Stopping after stage {stage}: RMS residual {eps_residual} "
                f"is at or below residual_tol={residual_tol}."
            )
            break

        stats_num_samples = _expanded_num_samples(num_samples_for_epsilon, net.in_size)
        _warn_if_frequency_underresolved(
            kappa,
            stats_num_samples,
            frequency_samples_per_mode,
            stage,
            chebyshev=chebyshev,
        )
        next_stage_x = x_stage2 if stage == 0 else x
        _warn_if_training_samples_underresolved(
            kappa,
            next_stage_x,
            frequency_samples_per_mode,
            stage,
            chebyshev=chebyshev,
        )

        params = _stage_correction_params_or_none(
            getattr(net, "_params", None), stage_correction_param_map
        )

        key, subkey = jax.random.split(key)
        net, net_kwargs_for_save = _build_next_stage(
            net,
            params,
            subkey,
            activation,
            eps_prediction,
            kappa,
            width_size=width_size,
            depth=depth,
            chebyshev=chebyshev,
            feature_map=feature_map,
            extra_stage_corrections=extra_stage_corrections,
            stage=stage,
            eps_residual=eps_residual,
        )

        residual_fun = residual_fun_s2
        loss_fun_base = loss_fun_s2
        loss_components_fun = loss_components_fun_s2
        loss_residual_components_fun = loss_residual_components_fun_s2
        gamma = gamma_s2
        gamma_g = gamma_g_s2
        # Next stages will use same as stage 2 data currently.
        training_samples = training_samples_stage2
        x = x_stage2

    return net
