import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from matplotlib import pyplot as plt

from multistage._utils import generate_data, stats, stats_chebyshev


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
    plt.figure()
    sc = plt.scatter(time, space, c=u_data, cmap="viridis", s=2, alpha=0.6)
    plt.colorbar(sc, label="u(t, x)")
    plt.xlabel("Dimension 0 (e.g., t)")
    plt.ylabel("Dimension 1 (e.g., x)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()
