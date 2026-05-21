import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from matplotlib import pyplot as plt

from multistage._utils import (
    _operator_scale,
    adaptive_sample,
    generate_concentrated_data,
    generate_data,
    stats,
    stats_chebyshev,
)


class _DummyModel(eqx.Module):
    """A dummy model to hold attributes required by stats()."""

    lb: jax.Array
    ub: jax.Array
    in_size: int
    out_size: int

    def __init__(self, lb, ub, in_size, out_size=1):
        self.lb = jnp.atleast_1d(lb)
        self.ub = jnp.atleast_1d(ub)
        self.in_size = in_size
        self.out_size = out_size


def test_operator_scale_uses_separate_pde_terms():
    """A PDE with u_t and u_xx should not be treated as u_txx."""
    kappa = jnp.array([2.0, 3.0])
    lb = jnp.array([0.0, -1.0])
    ub = jnp.array([1.0, 1.0])

    # Physical angular frequencies are [4, 3]. For separate terms with orders
    # (1, 0) and (0, 2), the dominant scale is 3**2. For a mixed derivative
    # term (1, 2), the scale is 4 * 3**2.
    np.testing.assert_allclose(_operator_scale(kappa, lb, ub, order=(1, 2)), 9.0)
    np.testing.assert_allclose(_operator_scale(kappa, lb, ub, order=((1, 2),)), 36.0)
    np.testing.assert_allclose(
        _operator_scale(kappa, lb, ub, order=(1, 2), beta=jnp.array([1.0, 0.25])),
        4.0,
    )


def test_epsilon_estimate_uses_physical_coordinate_scaling():
    """Stats must include the chain-rule factor from normalized coordinates."""
    net = _DummyModel(0.0, 1.0, in_size=1)
    params, static = eqx.partition(net, eqx.is_inexact_array)
    k = 40.0

    def residual_1d(model, x):
        z = 2.0 * (x - model.lb[0]) / (model.ub[0] - model.lb[0]) - 1.0
        residual = 2.0 * k * jnp.cos(k * z)
        return jnp.zeros_like(residual), residual

    eps_residual, eps_prediction, kappa = stats(
        params,
        static,
        residual_1d,
        num_samples=(512,),
        order=(1,),
        heuristic=1.0,
    )

    np.testing.assert_allclose(
        eps_prediction,
        eps_residual / (2.0 * kappa[0]),
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    "f_model, f_true, num_samples, heuristic",
    [(100, 120, 200, 0.92), (100, 120, 256, 0.92)],
)
def test_dominant_frequency_1d(f_model, f_true, num_samples, heuristic):
    """Test that kappa is computed correctly."""
    net = _DummyModel(-1.0, 1.0, in_size=1)
    params, static = eqx.partition(net, eqx.is_inexact_array)

    def residual_1d(model, x):
        pred = jnp.sin(f_model * x)
        true = jnp.sin(f_true * x)
        return pred, pred - true

    _, _, kappa = stats(
        params,
        static,
        residual_1d,
        num_samples=(num_samples,),
        order=(1,),
        heuristic=heuristic,
        frequency_estimator="zero_crossing",
    )
    np.testing.assert_allclose(
        kappa / heuristic,
        np.abs(f_model - f_true) / 2 + np.abs(f_model + f_true) / 2,
        err_msg="max frequency is wrong",
        atol=2,
    )
    np.testing.assert_allclose(
        kappa,
        np.abs(f_model + f_true) / 2,
        err_msg="dominant frequency is wrong due to poor heuristic choice",
        rtol=3e-2,
    )


def test_spectral_frequency_ignores_dc_offset():
    """The default Fourier estimator should not rely on sign changes."""
    net = _DummyModel(-1.0, 1.0, in_size=1)
    params, static = eqx.partition(net, eqx.is_inexact_array)
    k = 17 * jnp.pi

    def residual_1d(model, x):
        residual = 10.0 + 0.01 * jnp.sin(k * x)
        return residual, residual

    _, _, kappa = stats(
        params,
        static,
        residual_1d,
        num_samples=(256,),
        order=(1,),
    )
    np.testing.assert_allclose(kappa, jnp.array([k]), rtol=1e-12)


def test_spectral_frequency_uses_one_sided_amplitudes():
    """Interior Fourier modes should not be underweighted against Nyquist."""
    net = _DummyModel(-1.0, 1.0, in_size=1)
    params, static = eqx.partition(net, eqx.is_inexact_array)

    def residual_1d(model, x):
        del model
        interior = jnp.sin(4 * jnp.pi * x)
        nyquist = 0.75 * jnp.cos(16 * jnp.pi * x)
        residual = interior + nyquist
        return residual, residual

    _, _, kappa = stats(
        params,
        static,
        residual_1d,
        num_samples=(32,),
        order=(1,),
    )
    np.testing.assert_allclose(kappa, jnp.array([4 * jnp.pi]), rtol=1e-12)


@pytest.mark.parametrize(
    "f_model, f_true, num_samples",
    [(100, 120, 119), (100, 120, 256)],
)
def test_dominant_frequency_1d_chebyshev(f_model, f_true, num_samples):
    """Test that kappa is computed correctly."""
    net = _DummyModel(-1.0, 1.0, in_size=1)
    params, static = eqx.partition(net, eqx.is_inexact_array)

    def residual_1d(model, x):
        pred = jnp.sin(f_model * x)
        true = jnp.sin(f_true * x)
        return pred, pred - true

    _, _, kappa = stats_chebyshev(
        params,
        static,
        residual_1d,
        num_samples=(num_samples,),
        order=(1,),
    )
    np.testing.assert_allclose(kappa, 97)


def test_chebyshev_frequency_has_positive_floor():
    """Constant residuals should not create zero-scale correction networks."""
    net = _DummyModel(-1.0, 1.0, in_size=1)
    params, static = eqx.partition(net, eqx.is_inexact_array)

    def residual_1d(model, x):
        residual = jnp.ones_like(x)
        return residual, residual

    eps_residual, eps_prediction, kappa = stats_chebyshev(
        params,
        static,
        residual_1d,
        num_samples=(64,),
        order=(1,),
    )
    np.testing.assert_allclose(kappa, jnp.array([1]))
    assert jnp.isfinite(eps_residual)
    assert jnp.isfinite(eps_prediction)


@pytest.mark.parametrize(
    "fx_m, fx_t, ft_m, ft_t, num_samples, heuristic",
    [
        (50, 60, 20, 24, 200, 0.92),
        (50, 60, 20, 24, 256, 0.92),
    ],
)
def test_dominant_frequency_2d(fx_m, fx_t, ft_m, ft_t, num_samples, heuristic):
    """Test that kappa is computed correctly for anisotropic 2D residues.

    Uses a plane wave residual sin(K_m . x) - sin(K_t . x).
    """
    lb = np.array([-1.0, -1.0])
    ub = np.array([1.0, 1.0])
    net = _DummyModel(lb, ub, in_size=2)
    params, static = eqx.partition(net, eqx.is_inexact_array)

    def residual_2d(model, x, t):
        pred = jnp.sin(fx_m * x + ft_m * t)
        true = jnp.sin(fx_t * x + ft_t * t)
        return pred, pred - true

    _, _, kappa = stats(
        params,
        static,
        residual_2d,
        num_samples=(num_samples, num_samples),
        order=(1,),
        heuristic=heuristic,
        frequency_estimator="zero_crossing",
    )
    np.testing.assert_allclose(
        kappa[0] / heuristic,
        np.abs(fx_m - fx_t) / 2 + np.abs(fx_m + fx_t) / 2,
        err_msg="max frequency x is wrong",
        atol=1,
    )
    np.testing.assert_allclose(
        kappa[0],
        np.abs(fx_m + fx_t) / 2,
        err_msg="dominant frequency x is wrong due to poor heuristic choice",
        rtol=5e-2,
    )
    np.testing.assert_allclose(
        kappa[1] / heuristic,
        np.abs(ft_m - ft_t) / 2 + np.abs(ft_m + ft_t) / 2,
        err_msg="max frequency t is wrong",
        atol=1,
    )
    np.testing.assert_allclose(
        kappa[1],
        np.abs(ft_m + ft_t) / 2,
        err_msg="dominant frequency t is wrong due to poor heuristic choice",
        rtol=5e-2,
    )


@jax.vmap
def u_true(t, x):
    """Assumes lambda_1 = 0."""
    LAMBDA_2_TRUE = 0.01 / jnp.pi
    return jnp.sin(jnp.pi * x) * jnp.exp(-LAMBDA_2_TRUE * jnp.pi**2 * t)


def test_generated_data(lb=jnp.array([0, -1]), ub=jnp.array([1, 1])):
    """Make sure training data is quasi-uniform."""
    in_size = 2
    num_points = 2000
    x, u_data, key = generate_data(num_points, lb, ub, in_size, u_true)
    time, space = x

    assert len(x) == in_size
    assert time.shape == space.shape == u_data.shape == (num_points,)
    np.testing.assert_array_less(lb[0] - 1e-12, time)
    np.testing.assert_array_less(time, ub[0] + 1e-12)
    np.testing.assert_array_less(lb[1] - 1e-12, space)
    np.testing.assert_array_less(space, ub[1] + 1e-12)
    np.testing.assert_allclose(jnp.mean(time), 0.5, atol=0.03)
    np.testing.assert_allclose(jnp.mean(space), 0.0, atol=0.04)

    plt.figure()
    sc = plt.scatter(time, space, c=u_data, cmap="viridis", s=2, alpha=0.6)
    plt.colorbar(sc, label="u(t, x)")
    plt.xlabel("Dimension 0 (e.g., t)")
    plt.ylabel("Dimension 1 (e.g., x)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.close()


def test_concentrated_data(lb=jnp.array([0, -1]), ub=jnp.array([1, 1])):
    """Make sure training data is concentrated."""
    in_size = 2
    num_points = 1000
    x, u_data, key = generate_concentrated_data(
        num_points, lb, ub, in_size, u_true, 1, center=0.0, scale=0.01
    )
    time, space = x

    assert x.shape == (in_size, num_points)
    assert u_data.shape == (num_points,)
    np.testing.assert_array_less(lb[0] - 1e-12, time)
    np.testing.assert_array_less(time, ub[0] + 1e-12)
    np.testing.assert_array_less(lb[1] - 1e-12, space)
    np.testing.assert_array_less(space, ub[1] + 1e-12)
    np.testing.assert_allclose(jnp.mean(space), 0.0, atol=0.002)
    assert jnp.std(space) < 0.012

    plt.figure()
    sc = plt.scatter(time, space, c=u_data, cmap="viridis", s=2, alpha=0.6)
    plt.colorbar(sc, label="u(t, x)")
    plt.xlabel("Dimension 0 (e.g., t)")
    plt.ylabel("Dimension 1 (e.g., x)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.close()


def test_adaptive_sample_default_normal_sample():
    """The documented bool default for normal_sample should be accepted."""
    net = _DummyModel(jnp.array([0.0, -1.0]), jnp.array([1.0, 1.0]), in_size=2)

    def residual_fun(model, t, x):
        residual = (t - 0.5) ** 2 + x**2
        return residual, residual

    x_new, key = adaptive_sample(
        net,
        residual_fun,
        in_size=2,
        n_candidates=32,
        n_selected=5,
        key=jax.random.PRNGKey(0),
    )

    assert len(x_new) == 2
    assert x_new[0].shape == (5,)
    assert x_new[1].shape == (5,)
    assert key.shape == (2,)
