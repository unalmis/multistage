"""Utility functions for multi-stage neural networks."""

import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from interpax_fft import cheb_from_dct, cheb_pts
from jax import jit, vmap
from jax.scipy.fft import dct
from optax import squared_error
from paramax import NonTrainable


def partition(net):
    """Partition network for training.

    Returns
    -------
    trainable, frozen, static

    """
    trainable, static = eqx.partition(net, eqx.is_inexact_array)
    trainable, frozen = eqx.partition(
        trainable, eqx.is_inexact_array, is_leaf=is_not_trainable
    )
    return trainable, frozen, static


def is_not_trainable(leaf):
    """Return True if leaf is frozen parameter."""
    return isinstance(leaf, NonTrainable)


def generate_data(num_samples, lb, ub, in_size, true_data_fun, noise_level=0, key=None):
    """Generates training data on random points with some noise.

    Parameters
    ----------
    lb : jax.Array
        Lower bounds of the domain [x1_min, ... x_i_min, ... x_n_min].
    ub : jax.Array
        Upper bounds of the domain [x1_max, ... x_i_max, ... x_n_max].
    in_size : int
        Number of dimensions of input.
        The input should to the network should be ``in_size`` arguments.
    out_size : int
        The output should have shape (out_size, )
    true_data_fun : callable
        Function that can generate training data matching the API
        of ``vmap(model)(*x)``.
    noise_level : float
        Noise to add to true data.
        Default is zero.

    Returns
    -------
    x : list[jnp.ndarray]
        Points on which training data was computed.
    data : jnp.ndarray
        Training data on x points.
    key : float
        New random key.

    """
    if key is None:
        key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, in_size + 1)
    noise_key = keys[0]
    keys = keys[1:]
    x = []
    for i in range(in_size):
        x.append(
            jax.random.uniform(keys[i], (num_samples,), minval=lb[i], maxval=ub[i])
        )
    clean_data = true_data_fun(*x)

    final_key, noise_key = jax.random.split(noise_key)
    noise = (
        noise_level
        * jnp.std(clean_data)
        * jax.random.normal(noise_key, clean_data.shape)
    )
    data = clean_data + noise
    return x, data, final_key


def generate_concentrated_data(
    num_samples,
    lb,
    ub,
    in_size,
    true_data_fun,
    axis,
    center,
    scale,
    key=None,
):
    """Generates data focused around a specific value on one axis."""
    if key is None:
        key = jax.random.PRNGKey(42)

    key, k1, k2 = jax.random.split(key, 3)

    x = jax.random.uniform(k1, (num_samples, in_size), minval=lb, maxval=ub).T
    gauss_vals = jnp.clip(
        center + scale * jax.random.normal(k2, (num_samples,)), lb[axis], ub[axis]
    ).T
    x = x.at[axis].set(gauss_vals)
    u = true_data_fun(*x)
    return x, u, key


def adaptive_sample(
    net,
    residual_fun,
    in_size,
    n_candidates=50000,
    n_selected=2000,
    key=None,
    mode="top_k",
    k=1.0,
    c=0.0,
    normal_sample=False,
    center=0.0,
    scale=0.1,
):
    """
    Selects collocation points based on PDE residuals using adaptive strategies.

    Residual-based Adaptive Refinement (RAR)
    or Residual-based Adaptive Distribution (RAD).

    Parameters
    ----------
    net : eqx.Module
        The neural network model to evaluate.
    residual_fun : callable
        A function that computes the PDE residual.
    in_size : int
        Number of input dimensions.
    n_candidates : int, optional
        The number of candidate points to generate and evaluate. Default is 50,000.
    n_selected : int, optional
        The number of points to select from the candidates. Default is 2,000.
    key : jax.random.PRNGKey, optional
        JAX random key for generating candidates and probabilistic sampling.
        If None, a default key is used.
    mode : str, optional
        The sampling strategy to use:
        - 'top_k': Select the ``n_selected`` points with the highest absolute residuals.
        - 'probabilistic': Samples points with probability proportional to the residual.
    k : float, optional
        Hyperparameter for 'probabilistic' mode. Controls the sharpness of the
        distribution (higher k focuses more on high error). Default is 1.0.
    c : float, optional
        Hyperparameter for 'probabilistic' mode. Controls the flatness of the
        distribution (higher c adds more uniform randomness). Default is 0.0.
    normal_sample : bool or list[bool]
        Whether to sample additional points from normal distribution around x = 0.

    Returns
    -------
    list[jax.Array]
        A list of arrays [x1, x2, ..., xn], where each array has shape (n_selected,),
        containing the coordinates of the adaptively sampled points.
    key

    """
    if key is None:
        key = jax.random.PRNGKey(0)
    if isinstance(normal_sample, bool):
        normal_sample = (normal_sample,) * in_size
    elif len(normal_sample) != in_size:
        raise ValueError("normal_sample must be a bool or have length in_size.")
    if n_selected > n_candidates:
        raise ValueError("n_selected must be less than or equal to n_candidates.")

    lb, ub = net.lb, net.ub

    keys = jax.random.split(key, 2 * in_size + 2)
    uniform_keys = keys[:in_size]
    normal_keys = keys[in_size : 2 * in_size]
    x_candidates = []
    for i in range(in_size):
        if normal_sample[i]:
            num_1 = n_candidates // 2
            num_2 = n_candidates - num_1
        else:
            num_1 = n_candidates
        x_i = jax.random.uniform(uniform_keys[i], (num_1,), minval=lb[i], maxval=ub[i])
        if normal_sample[i]:
            x_i = jnp.concatenate(
                [
                    x_i,
                    jnp.clip(
                        center + scale * jax.random.normal(normal_keys[i], (num_2,)),
                        lb[i],
                        ub[i],
                    ),
                ]
            )
        x_candidates.append(x_i)

    _, residuals = residual_fun(net, *x_candidates)
    residuals = jnp.asarray(residuals)
    if residuals.ndim > 1:
        residuals = residuals.reshape((residuals.shape[0], -1))
        residuals = jnp.linalg.norm(residuals, axis=1)
    else:
        residuals = jnp.abs(residuals)

    if mode == "top_k":
        _, top_indices = jax.lax.top_k(residuals, n_selected)

    elif mode == "probabilistic":
        eps = 1e-10
        w = (residuals + eps) ** k
        mean_w = jnp.mean(w)
        weights = w + c * mean_w
        probs = weights / jnp.sum(weights)
        top_indices = jax.random.choice(
            keys[-2],
            jnp.arange(residuals.shape[0]),
            shape=(n_selected,),
            p=probs,
            replace=False,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    x_new = [xi[top_indices] for xi in x_candidates]
    return x_new, keys[-1]


def train_test_split(x_list, data, key, split_ratio=0.5):
    """Splits data into training and testing sets.

    Parameters
    ----------
    x_list : list[jax.Array]
        List of coordinate arrays [t, x, ...].
    data : jax.Array
        The data values corresponding to coordinates.
    key : jax.random.PRNGKey
        Random key for shuffling.
    split_ratio : float
        Percentage of data to keep for training (0.0 to 1.0).
        Default is 0.5.

    Returns
    -------
    x_train, data_train, x_test, data_test

    """
    num_samples = data.shape[0]
    indices = jnp.arange(num_samples)
    shuffled_indices = jax.random.permutation(key, indices)

    split_idx = int(num_samples * split_ratio)
    train_idx = shuffled_indices[:split_idx]
    test_idx = shuffled_indices[split_idx:]

    x_train = [xi[train_idx] for xi in x_list]
    x_test = [xi[test_idx] for xi in x_list]
    train = data[train_idx]
    test = data[test_idx]

    return x_train, train, x_test, test


def print_errors(net, x_test, u_data_test, check_dim=0):
    """Print some error metrics.

    If check_dim > 0, enforces that u_pred and u_data have the expected dim.
    """
    print("------------ Solution error --------------")
    u_pred = vmap(net)(*x_test)
    assert u_pred.shape == u_data_test.shape
    if check_dim > 0:
        assert u_pred.ndim == check_dim

    u_rms_err = jnp.sqrt(squared_error(u_pred, u_data_test).mean())
    u_abs_err = jnp.abs(u_pred - u_data_test).mean()
    u_rel_err = jnp.linalg.norm(u_pred - u_data_test) / jnp.linalg.norm(u_data_test)
    print(f"RMS error         = {u_rms_err:.6e}")
    print(f"Abs mean error    = {u_abs_err:.6e}")
    print(f"L2 relative error = {u_rel_err:.6e}\n")


def rescale(x, lb, ub):
    """Rescale x in [lb, ub] to [-1, 1]."""
    return 2.0 * (x - lb) / (ub - lb) - 1.0


def count_params(model):
    """Return trainable parameter count in the model."""
    trainable = eqx.filter(model, eqx.is_inexact_array, is_leaf=is_not_trainable)
    count = sum(x.size for x in jax.tree_util.tree_leaves(trainable) if x is not None)
    return count


def _operator_orders(order, in_size):
    """Return derivative multi-indices for terms in the linearized operator."""
    order = np.asarray(order, dtype=int)
    if order.ndim == 0:
        order = np.asarray([order.item()])

    if order.ndim == 1:
        if order.size == 1:
            if in_size == 1:
                order = order.reshape(1, 1)
            else:
                order = np.eye(in_size, dtype=int) * order.item()
        elif order.size == in_size:
            order = np.diag(order)
        else:
            raise ValueError("1D order must have length 1 or in_size.")
    elif order.ndim == 2:
        if order.shape[1] != in_size:
            raise ValueError("2D order must have shape (num_terms, in_size).")
    else:
        raise ValueError("order must be a scalar, vector, or matrix of multi-indices.")

    return jnp.asarray(order)


def _beta_rms(beta, num_terms):
    """Return scalar or per-operator-term RMS coefficient magnitudes."""
    beta = jnp.asarray(beta)
    if beta.ndim > 0 and beta.shape[-1] == num_terms:
        beta = beta.reshape((-1, num_terms))
        return jnp.sqrt(jnp.square(beta).mean(axis=0))
    return jnp.sqrt(jnp.square(beta).mean())


def _operator_scale(kappa, lb, ub, order, beta=1.0):
    """Estimate the dominant derivative amplification for operator terms.

    ``kappa`` is the angular frequency in normalized coordinates. PDE residuals
    are evaluated in physical coordinates, so each derivative contributes the
    chain-rule factor ``2 / (ub - lb)``. The multistage estimate balances the
    PDE residual against the dominant term in the linearized operator.
    """
    kappa = jnp.asarray(kappa)
    lb = jnp.asarray(lb)
    ub = jnp.asarray(ub)
    physical_kappa = jnp.abs(2.0 * kappa / (ub - lb))
    orders = _operator_orders(order, kappa.size)
    term_scales = jnp.prod(physical_kappa[None, :] ** orders, axis=1)
    beta = jnp.asarray(beta)
    if beta.ndim > 0:
        if beta.shape[-1] != term_scales.size:
            raise ValueError("per-term beta must have one value per operator term.")
        return jnp.max(jnp.abs(beta) * term_scales)
    return jnp.abs(beta) * jnp.max(term_scales)


@functools.partial(
    jit,
    static_argnames=[
        "static",
        "residual_fun",
        "num_samples",
        "order",
        "beta_fun",
        "heuristic",
    ],
)
def stats(
    params,
    static,
    residual_fun,
    num_samples=(1024,),
    order=(1,),
    beta_fun=None,
    heuristic=0.9,
):
    """Return scalar magnitude (epsilon) and vector frequency (kappa) of the residual.

    Parameters
    ----------
    params : eqx.Module
        Non-static params for the network.
    static : eqx.Module
        Static params for the network.
    residual_fun : callable
        Function to compute PDE residual.
    num_samples : tuple[int]
        Number of samples in each direction.
        Default is 1024.
        Note that kappa will be at most half this value.
    order : tuple[int] or tuple[tuple[int]]
        Derivative orders in the dominant linearized PDE operator. A 1D tuple is
        interpreted as separate terms in each input direction; pass nested
        multi-indices for mixed derivative terms.
    beta_fun : callable
        Function to compute coefficient magnitudes for the derivative terms.
        It may return a scalar coefficient or an array whose last axis has one
        coefficient per operator term.
    heuristic : float
        The factor should be the best guess <= 1 such that
        dominant frequency = max frequency * heuristic

    Returns
    -------
    epsilon_residual : float
        The root mean squared equation residual.
    epsilon_prediction : float
        The estimate for the root mean square prediction error.
    kappa : jax.Array
        Shape (self.in_size, )
        Heuristic value for dominant frequency in each direction.
        Assumes the x in exp(i kappa x) is normalized to (-1, 1).

    """
    net = eqx.combine(params, static)
    in_size, out_size = net.in_size, net.out_size
    lb, ub = net.lb, net.ub

    if len(num_samples) == 1:
        num_samples = (num_samples[0],) * in_size

    mesh = (jnp.linspace(lb[i], ub[i], num_samples[i]) for i in range(in_size))
    mesh = tuple(map(jnp.ravel, jnp.meshgrid(*mesh, indexing="ij")))
    _, f = residual_fun(net, *mesh)
    eps_residual = jnp.sqrt(squared_error(f).mean())
    if beta_fun is None:
        beta = 1
    else:
        beta = beta_fun(net, *mesh)
        beta = _beta_rms(beta, _operator_orders(order, in_size).shape[0])

    f = f.reshape(*num_samples, out_size)

    # We support anisotropic frequency dependence in the input variables,
    # but we assume each dimension of the output of the network is isotropic in
    # kappa. I.e. with a network N: ℝᵃ → ℝᵇ, we assume kappa is anisotropic in
    # the domain and isotropic in the codomain.

    # If the previous stage network did its job then the low frequency components
    # will be near zero, so the number of zero crossings should correlate with
    # the dominant frequency.
    def compute_count(i):
        cross = f.swapaxes(0, i)
        cross = (cross[:-1] * cross[1:]) < 0
        return cross.sum(0).mean()

    cross_count = jnp.asarray([compute_count(i) for i in range(in_size)])
    assert cross_count.shape == (in_size,)
    kappa = jnp.floor(heuristic * jnp.pi * cross_count / 2 + 1)

    operator_scale = _operator_scale(kappa, lb, ub, order, beta=beta)
    eps_prediction = eps_residual / operator_scale
    return eps_residual, eps_prediction, kappa


@functools.partial(
    jit,
    static_argnames=["static", "residual_fun", "num_samples", "order", "beta_fun"],
)
def stats_chebyshev(
    params,
    static,
    residual_fun,
    num_samples=(1024,),
    order=(1,),
    beta_fun=None,
):
    """Return scalar magnitude (epsilon) and vector frequency (kappa) of the residual.

    Parameters
    ----------
    params : eqx.Module
        Non-static params for the network.
    static : eqx.Module
        Static params for the network.
    residual_fun : callable
        Function to compute PDE residual.
    num_samples : tuple[int]
        Number of samples in each direction.
        Default is 1024.
    order : tuple[int] or tuple[tuple[int]]
        Derivative orders in the dominant linearized PDE operator. A 1D tuple is
        interpreted as separate terms in each input direction; pass nested
        multi-indices for mixed derivative terms.
    beta_fun : callable
        Function to compute coefficient magnitudes for the derivative terms.
        It may return a scalar coefficient or an array whose last axis has one
        coefficient per operator term.

    Returns
    -------
    epsilon_residual : float
        The root mean squared equation residual.
    epsilon_prediction : float
        The estimate for the root mean square prediction error.
    kappa : jax.Array
        Shape (self.in_size, )
        Heuristic value for dominant frequency in each direction.
        Assumes the x in cos(kappa arccos x) is normalized to (-1, 1).

    """
    net = eqx.combine(params, static)
    in_size, out_size = net.in_size, net.out_size
    lb, ub = net.lb, net.ub

    if len(num_samples) == 1:
        num_samples = (num_samples[0],) * in_size

    mesh = (cheb_pts(num_samples[i], (lb[i], ub[i])) for i in range(in_size))
    mesh = tuple(map(jnp.ravel, jnp.meshgrid(*mesh, indexing="ij")))
    _, f = residual_fun(net, *mesh)
    eps_residual = jnp.sqrt(squared_error(f).mean())
    if beta_fun is None:
        beta = 1
    else:
        beta = beta_fun(net, *mesh)
        beta = _beta_rms(beta, _operator_orders(order, in_size).shape[0])

    f = f.reshape(*num_samples, out_size)

    # We support anisotropic frequency dependence in the input variables,
    # but we assume each dimension of the output of the network is isotropic in
    # kappa. I.e. with a network N: ℝᵃ → ℝᵇ, we assume kappa is anisotropic in
    # the domain and isotropic in the codomain.
    def compute_freq(i):
        cheb = cheb_from_dct(dct(f, type=2, axis=i), i)
        return jnp.argmax(jnp.abs(cheb), axis=i).mean()

    kappa = jnp.asarray([compute_freq(i) for i in range(in_size)])
    assert kappa.shape == (in_size,)
    kappa = jnp.maximum(jnp.floor(kappa), 1)
    # this is a heuristic anyway
    operator_scale = _operator_scale(kappa, lb, ub, order, beta=beta)
    eps_prediction = eps_residual / operator_scale
    return eps_residual, eps_prediction, kappa
