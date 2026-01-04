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

from multistage import save

from ._io_utils import _ParamContainer, checkpoint_manager
from ._utils import (
    adaptive_sample,
    is_not_trainable,
    partition,
    rescale,
    stats,
    stats_chebyshev,
)

config.update("jax_enable_x64", True)


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
        if params is not None:
            if not params_are_trainable:
                params = non_trainable(params)
            if not isinstance(params, _ParamContainer):
                params = _ParamContainer(params)
            self._params = params

    @property
    def params(self):
        """Params for this stage."""
        if not hasattr(self, "_params"):
            raise AttributeError
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
            x = x[0]
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
        Approximate dominant frequency of output in each direction.
        Shape (s1._mlp.in_size, )
    width_size : int
        Size of each hidden layer.
    depth : int
        The number of hidden layers, including the output layer.
    activation : callable
        The activation function for each hidden layer after the first.
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
    chebyshev : bool
        Whether the frequency ``kappa`` is associated with a Chebyshev feature
        mapping instead of Fourier. Default is False.
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
        **kwargs,
    ):
        s1 = non_trainable(s1)
        self._s1 = s1
        self._epsilon = non_trainable(epsilon)
        self._kappa = non_trainable(jnp.asarray(kappa))
        self._chebyshev = chebyshev
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
        if params is not None:
            if not params_are_trainable:
                params = non_trainable(params)
            if not isinstance(params, _ParamContainer):
                params = _ParamContainer(params)
            self._params = params

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
        if not hasattr(self, "_params"):
            raise AttributeError
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

        Parameters
        ----------
        args : tuple[jnp.ndarray]
            Input coordinates, e.g. (x, t) for 2D problem.

        """
        s1 = self.s1
        u0 = s1(*args)

        x = rescale(jnp.stack(args), s1.lb, s1.ub)
        # TODO: Make frequency mapping separable to avoid diagonal waves.
        if self._chebyshev:
            # Ensure x ∈ (-1, 1), i.e. where arccos is differentiable.
            eps = 1 - 1e2 * jnp.finfo(jnp.array(1.0).dtype).eps
            x = jnp.clip(x, -eps, eps)
            x = jnp.cos(self._first(self.kappa * jnp.arccos(x)))
        else:
            x = jnp.sin(self._first(self.kappa * x))
        x = self.epsilon * self._mlp(x)

        if self.out_size == 1:
            x = x[0]

        return u0 + x

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


def _is_multiple_or_last(step, multiple, last):
    return ((step % multiple) == 0) or (step == (last - 1))


def _fill_lask_k_buffer(last_k_loss, loss_history):
    n_restore = min(loss_history.size, last_k_loss.size)
    if n_restore > 0:
        steps = np.arange(loss_history.size - n_restore, loss_history.size)
        last_k_loss[steps % last_k_loss.size] = loss_history[-n_restore:]
    return last_k_loss


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
    # progress & reproducibility params
    return_loss_history=True,
    print_every=100,
    checkpoint_path=None,
    checkpoint_every=5000,
    callback=None,
    callback_every=1000,
    key=None,
    debug=False,
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
    return_loss_history : bool, optional
        If True, returns the loss history alongside the model.
    print_every : int, optional
        Logging frequency.
    checkpoint_path : str, optional
        Path for saving/restoring checkpoints.
    checkpoint_every : int, optional
        Frequency of checkpoints.
    callback : callable, optional
        Function called every ``callback_every`` steps
        with signature ``callback(net,step)``.
    callback_every : int, optional
        Frequency of callback execution.
    key : jax.random.PRNGKey, optional
        Random key for sampling.
    debug : bool, optional
        Whether to print additional details for testing and debugging.

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
    last_k_loss = np.zeros(print_every)

    manager = None
    start_step = 0
    if checkpoint_path:
        manager = checkpoint_manager(checkpoint_path)

        if manager.latest_step() is not None:
            start_step = manager.latest_step()
            print(f"\n=== Resuming training from step {start_step} ===")
            restored = {"trainable": trainable, "opt_state": opt_state}
            if return_loss_history:
                restored["loss_history"] = jnp.zeros(start_step + 1)

            try:
                restored = manager.restore(
                    start_step, args=ocp.args.StandardRestore(restored)
                )
            except ValueError:
                restored["loss_history"] = jnp.zeros(start_step + 2)
                restored = manager.restore(
                    start_step, args=ocp.args.StandardRestore(restored)
                )

            trainable = restored["trainable"]
            opt_state = restored["opt_state"]
            net = eqx.combine(trainable, frozen, static)

            if return_loss_history:
                loss_history = np.array(restored["loss_history"])
                last_k_loss = _fill_lask_k_buffer(last_k_loss, loss_history)
                loss_history = list(loss_history)

            start_step += 1

    if debug:
        print("\n-----   Static  -----")
        print(static)
        print("\n----- Trainable -----")
        print(trainable)
        print("\n-----   Frozen  -----")
        print(frozen)
        print("\n----- Recombine -----")
        print(eqx.combine(trainable, frozen, static))

    def loss(trainable, frozen, static, *args):
        net = eqx.combine(trainable, frozen, static)
        return loss_fun(net, *args)

    @partial(jit, static_argnames=["static"])
    def make_step(trainable, frozen, static, opt_state, *args):
        loss_value, grads = value_and_grad(loss)(trainable, frozen, static, *args)
        if is_lbfgs:

            def loss_lbfgs(trainable):
                return loss(trainable, frozen, static, *args)

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

    for step in range(start_step, steps):
        if (
            (adaptive_sampler is not None)
            and (step % adaptive_sample_freq == 0)
            and (step >= 999)
        ):
            print(f"Resampled at step {step}.")
            x_col, key = adaptive_sampler(
                eqx.combine(trainable, frozen, static), key=key
            )

        trainable, opt_state, loss_value = make_step(
            trainable, frozen, static, opt_state, *x, training_samples, *x_col
        )

        if return_loss_history:
            loss_history.append(loss_value)

        last_k_loss[step % print_every] = loss_value
        if _is_multiple_or_last(step, print_every, steps):
            last_k = loss_value if (step == 0) else last_k_loss.mean()
            print(
                f"Step {step}, Loss: {loss_value:.6e}, "
                f"Last {print_every} mean loss: {last_k}.",
                flush=True,
            )
            net = eqx.combine(trainable, frozen, static)
            net.print_params()
            if debug:
                net.print_frozen_params()

        if (
            manager
            and (step > start_step)
            and _is_multiple_or_last(step, checkpoint_every, steps)
        ):
            man_args = {"trainable": trainable, "opt_state": opt_state}
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
    learning_rate,
    adaptive_sampler=None,
    adaptive_sample_freq=100,
    # progress & reproducibility params
    checkpoint_path=None,
    checkpoint_every=100,
    callback=None,
    callback_every=100,
    key=None,
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
    learning_rate : float
        Stepsize/damping parameter for the solver.
    adaptive_sampler : callable, optional
        Function to generate new collocation points.
    adaptive_sample_freq : int, optional
        Frequency of resampling collocation points.
    checkpoint_path : str, optional
        Path for saving/restoring checkpoints.
    checkpoint_every : int, optional
        Frequency of checkpoints.
    callback : callable, optional
        Function called every ``callback_every`` steps
        with signature ``callback(net,step)``.
    callback_every : int, optional
        Frequency of callback execution.
    key : jax.random.PRNGKey, optional
        Random key for sampling.

    Returns
    -------
    net : eqx.Module
        The trained model.

    """
    trainable, frozen, static = partition(net)
    solver = optimistix.LevenbergMarquardt(
        rtol=rtol,
        atol=atol,
        verbose=frozenset({"step", "loss", "step_size"}),
        linear_solver=linear_solver,
    )

    manager = None
    start_step = 0
    if checkpoint_path:
        manager = checkpoint_manager(checkpoint_path)

        if manager.latest_step() is not None:
            start_step = manager.latest_step()
            print(f"\n=== Resuming training from step {start_step} ===")
            restored = manager.restore(
                start_step,
                args=ocp.args.StandardRestore({"trainable": trainable}),
            )
            trainable = restored["trainable"]
            net = eqx.combine(trainable, frozen, static)
            start_step += 1

    def loss(trainable, args):
        frozen, static, *rest = args
        net = eqx.combine(trainable, frozen, static)
        return loss_fun_unreduced(net, *rest)

    print(f"--- Values at step {start_step} ---")
    net.print_params()

    print("--- Starting training ---")
    start_train_time = time.time()

    x_col = [None] * net.in_size
    step_per_opt = min(adaptive_sample_freq, checkpoint_every, steps)

    for step in range(start_step, steps, step_per_opt):
        if (
            (adaptive_sampler is not None)
            and (step == start_step)
            and (
                step // adaptive_sample_freq
                > (step - step_per_opt) // adaptive_sample_freq
            )
        ):
            print(f"Resampled at step {step}.")
            x_col, key = adaptive_sampler(
                eqx.combine(trainable, frozen, static), key=key
            )

        sol = optimistix.least_squares(
            loss,
            solver,
            trainable,
            args=(frozen, static, *x, training_samples, *x_col),
            max_steps=step_per_opt,
            throw=False,
        )
        trainable = sol.value
        print(optimistix.RESULTS[sol.result])
        # can't extract loss history programmatically from optimistix solver:
        #    https://github.com/patrick-kidger/optimistix/issues/52

        net = eqx.combine(trainable, frozen, static)
        net.print_params()

        current_step = step + step_per_opt
        if manager and (
            (current_step // checkpoint_every > step // checkpoint_every)
            or (current_step == steps)
        ):
            manager.save(
                current_step,
                args=ocp.args.StandardSave({"trainable": trainable}),
            )

        if callback and (
            (current_step // callback_every > step // callback_every)
            or (current_step == steps)
        ):
            current_net = eqx.combine(trainable, frozen, static)
            callback(current_net, step)

    end_train_time = time.time()
    print("--- Finished training ---\n")
    print(f"Training time: {end_train_time - start_train_time:.2f}")

    if manager:
        manager.wait_until_finished()

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
    chebyshev=False,
    x_stage2=None,
    training_samples_stage2=None,
    # progress & reproducibility params
    return_loss_history=True,
    print_every=100,
    key=None,
    net_kwargs_for_save=None,
    name="",
    checkpoint_dir="checkpoints",
    checkpoint_every=5000,
    benchmark_state=None,
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
        Order of error estimation.
    beta_fun : callable, optional
        Function defining the beta distribution for error bounds.
    heuristic : float, optional
        Heuristic multiplier for error estimation. Default is 0.9.
    chebyshev : bool
        Whether to use Chebyshev feature mapping instead of Fourier.
        If given, ``heuristic`` is ignored.
    x_stage2 : tuple of jax.Array, optional
        Input coordinates for stage 2 and beyond. Default is ``x``.
    training_samples_stage2 : jax.Array, optional
        Training data for stage 2 and beyond. Default is ``training_samples``.
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
        Directory to store stage-specific checkpoints.
    checkpoint_every : int, optional
        Frequency of checkpointing within stages.
    benchmark_state : callable, optional
        Callback for external benchmarking or logging. Signature:
        ``benchmark_state(net,stage,name,step=step)``.

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
    if training_samples_stage2 is None:
        x_stage2 = x
        training_samples_stage2 = training_samples

    residual_fun = residual_fun_s1
    loss_fun = loss_fun_s1
    loss_histories = []

    for stage in range(n_stages):
        current_callback = None
        if benchmark_state is not None:

            def current_callback(n, s):  # noqa: F811
                benchmark_state(n, stage, name, step=s)

        if adaptive_sample_freq > 0:
            adaptive_sampler = partial(
                adaptive_sample,
                residual_fun=residual_fun,
                in_size=net.in_size,
                n_candidates=len(x[0]) * 10,
                n_selected=len(x[0]) // 2,
            )
        else:
            adaptive_sampler = None

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
            return_loss_history=return_loss_history,
            print_every=print_every,
            checkpoint_path=os.path.join(checkpoint_dir, f"{name}_stage_{stage}"),
            checkpoint_every=checkpoint_every,
            callback=current_callback,
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
            **({} if chebyshev else {"heuristic": heuristic}),
        )
        print(f"Stage {stage} statistics:")
        print(f"RMS residual estimate used for stage {stage +1} is {eps_residual}.")
        print(f"RMS prediction residual used for stage {stage +1} is {eps_prediction}.")
        print(f"Estimate frequency kappa used for stage {stage +1} is {kappa}.")

        params = getattr(net, "_params", None)
        if params is not None:
            params = eqx.filter(
                params, is_not_trainable, inverse=True, is_leaf=is_not_trainable
            )

        key, subkey = jax.random.split(key)
        net_kwargs_for_save = dict(
            epsilon=eps_prediction,
            kappa=kappa,
            width_size=width_size,
            depth=depth,
            params_are_trainable=params is not None,
            chebyshev=chebyshev,
        )
        net = Stage2(
            net, params=params, key=subkey, activation=activation, **net_kwargs_for_save
        )

        residual_fun = residual_fun_s2
        loss_fun = loss_fun_s2
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
    linear_solver=(lx.QR(), lx.NormalCG(rtol=1e-7, atol=1e-7)),
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
    chebyshev=False,
    x_stage2=None,
    training_samples_stage2=None,
    # Progress & reproducibility params
    print_every=100,
    key=None,
    net_kwargs_for_save=None,
    name="",
    checkpoint_dir="checkpoints",
    checkpoint_every=100,
    benchmark_state=None,
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
        The initial damping parameter (step size) for the Levenberg-Marquardt
        algorithm.
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
        Order of error estimation.
    beta_fun : callable, optional
        Function defining the beta distribution for error bounds.
    heuristic : float, optional
        Heuristic multiplier for error estimation. Default is 0.9.
    chebyshev : bool
        Whether to use Chebyshev feature mapping instead of Fourier.
        If given, ``heuristic`` is ignored.
    x_stage2 : tuple of jax.Array, optional
        Input coordinates for stage 2 and beyond. Default is ``x``.
    training_samples_stage2 : jax.Array, optional
        Training data for stage 2 and beyond. Default is ``training_samples``.
    print_every : int, optional
        Logging frequency.
    key : jax.random.PRNGKey, optional
        Random key for initialization and sampling.
    net_kwargs_for_save : dict, optional
        Additional metadata to save with the model.
    name : str, optional
        Base name for saving models and checkpoints.
    checkpoint_dir : str, optional
        Directory to store stage-specific checkpoints.
    checkpoint_every : int, optional
        Frequency of checkpointing within stages. Default is 100.
        LBFGS warmup steps will checkpoint with 10 times less frequency.
    benchmark_state : callable, optional
        Callback for external benchmarking or logging. Signature:
        ``benchmark_state(net,stage,name,step=step)``.

    Returns
    -------
    net : eqx.Module
        The final trained multi-stage model.

    """
    if key is None:
        key = jax.random.PRNGKey(42)
    if net_kwargs_for_save is None:
        net_kwargs_for_save = {}
    if training_samples_stage2 is None:
        x_stage2 = x
        training_samples_stage2 = training_samples

    residual_fun = residual_fun_s1
    loss_fun = loss_fun_s1
    loss_fun_unreduced = loss_fun_s1_unreduced
    linear_solver, linear_solver_next = linear_solver

    for stage in range(n_stages):
        current_callback = None
        if benchmark_state is not None:

            def current_callback(n, s):  # noqa: F811
                benchmark_state(n, stage, name, step=s)

        if adaptive_sample_freq > 0:
            adaptive_sampler = partial(
                adaptive_sample,
                residual_fun=residual_fun,
                in_size=net.in_size,
                n_candidates=len(x[0]) * 10,
                n_selected=len(x[0]) // 2,
            )
        else:
            adaptive_sampler = None

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
            return_loss_history=True,
            print_every=print_every,
            checkpoint_path=os.path.join(
                checkpoint_dir, f"{name}_lbfgs_warmup_stage_{stage}"
            ),
            checkpoint_every=checkpoint_every * 10,
            callback=current_callback,
        )

        net = _trust_region_train(
            net=net,
            loss_fun_unreduced=loss_fun_unreduced,
            x=x,
            training_samples=training_samples,
            steps=steps,
            rtol=rtol,
            atol=atol,
            linear_solver=linear_solver,
            learning_rate=learning_rate,
            adaptive_sampler=adaptive_sampler,
            adaptive_sample_freq=adaptive_sample_freq,
            checkpoint_path=os.path.join(checkpoint_dir, f"{name}_stage_{stage}"),
            checkpoint_every=checkpoint_every,
            callback=current_callback,
        )

        linear_solver = linear_solver_next
        loss_fun_unreduced = loss_fun_s2_unreduced
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
            **({} if chebyshev else {"heuristic": heuristic}),
        )
        print(f"Stage {stage} statistics:")
        print(f"RMS residual estimate used for stage {stage +1} is {eps_residual}.")
        print(f"RMS prediction residual used for stage {stage +1} is {eps_prediction}.")
        print(f"Estimate frequency kappa used for stage {stage +1} is {kappa}.")

        params = getattr(net, "_params", None)
        if params is not None:
            params = eqx.filter(
                params, is_not_trainable, inverse=True, is_leaf=is_not_trainable
            )

        key, subkey = jax.random.split(key)
        net_kwargs_for_save = dict(
            epsilon=eps_prediction,
            kappa=kappa,
            width_size=width_size,
            depth=depth,
            params_are_trainable=params is not None,
            chebyshev=chebyshev,
        )
        net = Stage2(
            net, params=params, key=subkey, activation=activation, **net_kwargs_for_save
        )

        residual_fun = residual_fun_s2
        loss_fun = loss_fun_s2
        # Next stages will use same as stage 2 data currently.
        training_samples = training_samples_stage2
        x = x_stage2

    return net
