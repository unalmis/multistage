"""Test multi-stage network on the the 2D Burger's equation inverse problem."""

from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pytest
from jax import config, grad, vmap
from optax import squared_error
from orthax.hermite import hermgauss

from multistage import (
    Stage1,
    Stage2,
    multistage_train,
    multistage_trust_region_train,
    plot_2d_residual,
    plot_loss,
)
from multistage._multistage import _stage_correction_params_or_none, _train
from multistage._utils import (
    count_params,
    generate_concentrated_data,
    generate_data,
    print_errors,
    train_test_split,
)

config.update("jax_enable_x64", True)

# Domain bounds [t, x]
lb = jnp.array([0.0, -1.0])
ub = jnp.array([1.0, 1.0])


u_fn_s1 = Stage1.__call__
u_t_s1 = grad(u_fn_s1, argnums=1)
u_x_s1 = grad(u_fn_s1, argnums=2)
u_xx_s1 = grad(grad(u_fn_s1, argnums=2), argnums=2)


@partial(vmap, in_axes=(None, 0, 0))
def residual_s1_burgers(model, t, x):
    """Calculates the PDE residual for the Stage 1 model.

    Parameters
    ----------
    model : Stage1
        The Stage 1 PINN model.
    t : jax.Array
        Time coordinate.
    x : jax.Array
        Spatial coordinate.
    stage : int
        Stage of the model.
        Must be 1.

    Returns
    -------
    u, f tuple[jax.Array]
        The predicted values for u and f.

    """
    u = model(t, x)
    ut = u_t_s1(model, t, x)
    ux = u_x_s1(model, t, x)
    uxx = u_xx_s1(model, t, x)
    lam_1, lam_2 = total_lambdas(model)
    f = ut + lam_1 * u * ux - lam_2 * uxx
    return u, f


def loss_s1_burgers(model, t_data, x_data, u_data, t_col=None, x_col=None, gamma=0.5):
    """Calculates the total loss for the Stage 1 model.

    Parameters
    ----------
    model : Stage1
        The Stage 1 PINN model.
    t_data : jax.Array
        Batch of time coordinates.
    x_data : jax.Array
        Batch of spatial coordinates.
    u_data : jax.Array
        Batch of noisy training data for u.
    t_col : jax.Array
        Time samples where only PDE residual is computed.
    x_col : jax.Array
        X samples where only PDE residual is computed.

    Returns
    -------
    loss : float
        The total loss (data_loss + residual_loss).

    """
    u_pred, f_pred = residual_s1_burgers(model, t_data, x_data)
    loss_data = squared_error(u_pred, u_data).mean()
    loss_resid = f_pred.ravel()

    if t_col is not None and x_col is not None:
        _, col_resid = residual_s1_burgers(model, t_col, x_col)
        loss_resid = jnp.concatenate((loss_resid, col_resid.ravel()))

    return (1 - gamma) * loss_data + gamma * squared_error(loss_resid).mean()


def loss_s1_burgers_unreduced(
    model, t_data, x_data, u_data, t_col=None, x_col=None, gamma=0.5
):
    """Returns loss residual vector for use with least squares optimizers.

    Parameters
    ----------
    model : Stage1
        The Stage 1 PINN model.
    t_data : jax.Array
        Batch of time coordinates.
    x_data : jax.Array
        Batch of spatial coordinates.
    u_data : jax.Array
        Batch of noisy training data for u.
    t_col : jax.Array
        Time samples where only PDE residual is computed.
    x_col : jax.Array
        X samples where only PDE residual is computed.

    Returns
    -------
    jax.Array
        The total loss (data_loss + residual_loss).

    """
    u_pred, f_pred = residual_s1_burgers(model, t_data, x_data)
    loss_data = jnp.ravel(u_pred - u_data)
    loss_resid = f_pred.ravel()

    if t_col is not None and x_col is not None:
        _, col_resid = residual_s1_burgers(model, t_col, x_col)
        col_resid = col_resid.ravel()
        loss_resid = jnp.concatenate((loss_resid, col_resid))

    loss_data *= jnp.sqrt((1 - gamma) / loss_data.size)
    loss_resid *= jnp.sqrt(gamma / loss_resid.size)

    return jnp.concatenate((loss_data, loss_resid))


def _scalar_param(x):
    if hasattr(x, "ndim") and x.ndim > 0:
        return x[0]
    return x


def _stage_lambda_2(model):
    lam_2 = model.get_param("lambda_2", None)
    log_lam_2 = model.get_param("log_lambda_2", None)
    if lam_2 is not None and log_lam_2 is not None:
        raise ValueError(
            "A single stage cannot define both 'lambda_2' and 'log_lambda_2'."
        )
    if lam_2 is not None:
        return lam_2
    if log_lam_2 is not None:
        return jnp.exp(log_lam_2)
    return 0.0


def u_prev_fn_s2(model, t, x):
    return model.s1(t, x)


u_prev_t_s2 = grad(u_prev_fn_s2, argnums=1)
u_prev_x_s2 = grad(u_prev_fn_s2, argnums=2)
u_prev_xx_s2 = grad(grad(u_prev_fn_s2, argnums=2), argnums=2)
u_corr_fn_s2 = Stage2.compute_s2
u_corr_t_s2 = grad(u_corr_fn_s2, argnums=1)
u_corr_xx_s2 = grad(grad(u_corr_fn_s2, argnums=2), argnums=2)


@partial(grad, argnums=2)
def _u_prod_x(model, t, x):
    # Linearized derivative of the nonlinear Burgers flux cross term.
    return model.s1(t, x) * model.compute_s2(t, x)


@partial(vmap, in_axes=(None, 0, 0))
def residual_s2_burgers(model, t, x):
    """Calculates the PDE residual for the Stage 2 model.

    Parameters
    ----------
    model : Stage2
        The Stage 2 PINN model.
    t : jax.Array
        Time coordinate.
    x : jax.Array
        Spatial coordinate.

    Returns
    -------
    u, f tuple[jax.Array]
        The predicted values for u and f.

    """
    u = model(t, x)
    u_prev = model.s1(t, x)
    u_prev_t = u_prev_t_s2(model, t, x)
    u_prev_x = u_prev_x_s2(model, t, x)
    u_prev_xx = u_prev_xx_s2(model, t, x)
    u_corr_t = u_corr_t_s2(model, t, x)
    u_corr_xx = u_corr_xx_s2(model, t, x)
    lam_1, lam_2 = total_lambdas(model.s1)
    dlam_1 = _scalar_param(model.get_param("lambda_1", 0.0))
    dlam_2 = _scalar_param(_stage_lambda_2(model))

    prev_resid = u_prev_t + lam_1 * u_prev * u_prev_x - lam_2 * u_prev_xx
    corr_resid = (
        u_corr_t
        + lam_1 * _u_prod_x(model, t, x)
        - lam_2 * u_corr_xx
        + dlam_1 * u_prev * u_prev_x
        - dlam_2 * u_prev_xx
    )
    f = prev_resid + model.epsilon * corr_resid
    return u, f


def total_lambdas(model):

    epsilons = [model.epsilon]
    lams_1 = [model.get_param("lambda_1", 0.0)]
    lams_2 = [_stage_lambda_2(model)]
    s1 = model
    while hasattr(s1, "s1"):
        s1 = s1.s1
        lams_1.append(s1.get_param("lambda_1", 0.0))
        lams_2.append(_stage_lambda_2(s1))
        epsilons.append(s1.epsilon)

    lam_1 = 0.0
    lam_2 = 0.0
    for epsilon, lam_1_i, lam_2_i in zip(epsilons, lams_1, lams_2):
        lam_1 += epsilon * lam_1_i
        lam_2 += epsilon * lam_2_i

    return _scalar_param(lam_1), _scalar_param(lam_2)


def test_total_lambdas_for_forward_and_inverse_stages():
    """Forward params stay fixed; inverse params are staged corrections."""
    lb_test = jnp.array([0.0, -1.0])
    ub_test = jnp.array([1.0, 1.0])
    s1 = Stage1(
        lb_test,
        ub_test,
        in_size=2,
        out_size=1,
        params={
            "lambda_1": jnp.array([1.0]),
            "log_lambda_2": jnp.log(jnp.array([0.25])),
        },
        params_are_trainable=False,
    )
    s2 = Stage2(s1, epsilon=1e-3, kappa=jnp.array([2.0, 3.0]))
    lam_1, lam_2 = total_lambdas(s2)
    assert jnp.allclose(lam_1, 1.0)
    assert jnp.allclose(lam_2, 0.25)

    s2_inverse = Stage2(
        s1,
        epsilon=1e-3,
        kappa=jnp.array([2.0, 3.0]),
        params={
            "lambda_1": jnp.array([2.0]),
            "lambda_2": jnp.array([0.5]),
        },
        params_are_trainable=True,
    )
    lam_1, lam_2 = total_lambdas(s2_inverse)
    assert jnp.allclose(lam_1, 1.0 + 1e-3 * 2.0)
    assert jnp.allclose(lam_2, 0.25 + 1e-3 * 0.5)

    s2_log_inverse = Stage2(
        s1,
        epsilon=1e-3,
        kappa=jnp.array([2.0, 3.0]),
        params={"log_lambda_2": jnp.log(jnp.array([0.5]))},
        params_are_trainable=True,
    )
    lam_1, lam_2 = total_lambdas(s2_log_inverse)
    assert jnp.allclose(lam_1, 1.0)
    assert jnp.allclose(lam_2, 0.25 + 1e-3 * 0.5)

    s2_partial_inverse = Stage2(
        s1,
        epsilon=1e-3,
        kappa=jnp.array([2.0, 3.0]),
        params={"lambda_1": jnp.array([3.0])},
        params_are_trainable=True,
    )
    lam_1, lam_2 = total_lambdas(s2_partial_inverse)
    assert jnp.allclose(lam_1, 1.0 + 1e-3 * 3.0)
    assert jnp.allclose(lam_2, 0.25)

    s2_signed_diffusion_correction = Stage2(
        s1,
        epsilon=1e-3,
        kappa=jnp.array([2.0, 3.0]),
        params={"lambda_2": jnp.array([-0.5])},
        params_are_trainable=True,
    )
    lam_1, lam_2 = total_lambdas(s2_signed_diffusion_correction)
    assert jnp.allclose(lam_1, 1.0)
    assert jnp.allclose(lam_2, 0.25 - 1e-3 * 0.5)


def test_total_lambdas_rejects_ambiguous_lambda_2_parameterization():
    """Physical and transformed diffusion params cannot coexist on one stage."""
    s1 = Stage1(
        lb,
        ub,
        in_size=2,
        out_size=1,
        params={
            "lambda_1": jnp.array([1.0]),
            "log_lambda_2": jnp.log(jnp.array([0.25])),
        },
        params_are_trainable=False,
    )
    s2 = Stage2(
        s1,
        epsilon=1e-3,
        kappa=jnp.array([2.0, 3.0]),
        params={
            "lambda_2": jnp.array([0.5]),
            "log_lambda_2": jnp.log(jnp.array([0.5])),
        },
        params_are_trainable=True,
    )

    with pytest.raises(ValueError, match="lambda_2.*log_lambda_2"):
        total_lambdas(s2)


def test_forward_problem_params_stay_fixed_across_stage_training():
    """Frozen PDE params stay fixed and do not seed correction params."""
    key = jax.random.PRNGKey(0)
    s1_key, s2_key = jax.random.split(key)
    lam_1 = jnp.array([0.3])
    lam_2 = jnp.array([0.2])
    s1 = Stage1(
        lb,
        ub,
        in_size=2,
        out_size=1,
        width_size=4,
        depth=2,
        params={"lambda_1": lam_1, "log_lambda_2": jnp.log(lam_2)},
        params_are_trainable=False,
        key=s1_key,
    )
    t = jnp.linspace(0.0, 1.0, 8)
    x = jnp.linspace(-1.0, 1.0, 8)
    u_stage1 = jnp.linspace(-0.5, 0.5, 8)
    initial_s1_output = vmap(s1)(t, x)

    def stage1_loss_fun(model, t_data, x_data, u_data, t_col=None, x_col=None):
        del t_col, x_col
        u_loss = jnp.mean((vmap(model)(t_data, x_data) - u_data) ** 2)
        current_lam_1, current_lam_2 = total_lambdas(model)
        param_loss = (current_lam_1 - 5.0) ** 2 + (current_lam_2 - 7.0) ** 2
        return u_loss + param_loss

    trained_s1 = _train(
        s1,
        stage1_loss_fun,
        [t, x],
        u_stage1,
        optimizer=optax.sgd,
        steps=5,
        learning_rate=1e-2,
        adaptive_sampler=None,
        return_loss_history=False,
        print_every=0,
        checkpoint_path=None,
        normalize_loss=False,
    )

    trained_s1_output = vmap(trained_s1)(t, x)
    assert not jnp.allclose(trained_s1_output, initial_s1_output)
    np.testing.assert_allclose(trained_s1.get_param("lambda_1"), lam_1, rtol=0, atol=0)
    np.testing.assert_allclose(
        trained_s1.get_param("log_lambda_2"), jnp.log(lam_2), rtol=0, atol=0
    )

    correction_params = _stage_correction_params_or_none(trained_s1._params)
    assert correction_params is None
    s2 = Stage2(
        trained_s1,
        epsilon=0.1,
        kappa=jnp.array([2.0, 3.0]),
        width_size=4,
        depth=2,
        params=correction_params,
        params_are_trainable=correction_params is not None,
        key=s2_key,
    )
    assert tuple(s2.params.items()) == ()
    np.testing.assert_allclose(
        total_lambdas(s2), total_lambdas(trained_s1), rtol=0, atol=0
    )

    u_stage2 = jnp.ones_like(u_stage1)
    initial_s2_correction = vmap(s2.compute_s2)(t, x)

    def stage2_loss_fun(model, t_data, x_data, u_data, t_col=None, x_col=None):
        del t_col, x_col
        correction_loss = jnp.mean(
            (vmap(model.compute_s2)(t_data, x_data) - u_data) ** 2
        )
        current_lam_1, current_lam_2 = total_lambdas(model)
        param_loss = (current_lam_1 - 5.0) ** 2 + (current_lam_2 - 7.0) ** 2
        return correction_loss + param_loss

    trained_s2 = _train(
        s2,
        stage2_loss_fun,
        [t, x],
        u_stage2,
        optimizer=optax.sgd,
        steps=5,
        learning_rate=1e-2,
        adaptive_sampler=None,
        return_loss_history=False,
        print_every=0,
        checkpoint_path=None,
        normalize_loss=False,
    )

    assert not jnp.allclose(vmap(trained_s2.compute_s2)(t, x), initial_s2_correction)
    assert tuple(trained_s2.params.items()) == ()
    np.testing.assert_allclose(
        trained_s2.s1.get_param("lambda_1"),
        trained_s1.get_param("lambda_1"),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        trained_s2.s1.get_param("log_lambda_2"),
        trained_s1.get_param("log_lambda_2"),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        total_lambdas(trained_s2), total_lambdas(trained_s1), rtol=0, atol=0
    )


def test_log_parameterized_inverse_params_seed_physical_zero_corrections():
    """A log base parameter should become a signed zero physical correction."""
    key = jax.random.PRNGKey(0)
    s1_key, s2_key = jax.random.split(key)
    base_lam_1 = jnp.array([0.3])
    base_lam_2 = jnp.array([0.2])
    s1 = Stage1(
        lb,
        ub,
        in_size=2,
        out_size=1,
        width_size=4,
        depth=2,
        params={"lambda_1": base_lam_1, "log_lambda_2": jnp.log(base_lam_2)},
        params_are_trainable=True,
        key=s1_key,
    )
    correction_params = _stage_correction_params_or_none(s1._params)

    assert correction_params.keys == ("lambda_1", "lambda_2")
    np.testing.assert_allclose(correction_params["lambda_1"], jnp.array([0.0]))
    np.testing.assert_allclose(correction_params["lambda_2"], jnp.array([0.0]))

    s2 = Stage2(
        s1,
        epsilon=0.1,
        kappa=jnp.array([2.0, 3.0]),
        width_size=4,
        depth=2,
        params=correction_params,
        params_are_trainable=True,
        key=s2_key,
    )
    np.testing.assert_allclose(total_lambdas(s2), total_lambdas(s1), rtol=0, atol=0)

    t = jnp.linspace(0.0, 1.0, 8)
    x = jnp.linspace(-1.0, 1.0, 8)
    u_data = jnp.zeros(8)

    def stage2_param_loss(model, t_data, x_data, u_data, t_col=None, x_col=None):
        del t_data, x_data, u_data, t_col, x_col
        return jnp.mean((model.get_param("lambda_2") + 0.5) ** 2)

    trained_s2 = _train(
        s2,
        stage2_param_loss,
        [t, x],
        u_data,
        optimizer=optax.sgd,
        steps=5,
        learning_rate=1e-1,
        adaptive_sampler=None,
        return_loss_history=False,
        print_every=0,
        checkpoint_path=None,
        normalize_loss=False,
    )

    assert trained_s2.get_param("log_lambda_2", None) is None
    assert trained_s2.get_param("lambda_2")[0] < 0.0
    lam_1_total, lam_2_total = total_lambdas(trained_s2)
    np.testing.assert_allclose(lam_1_total, base_lam_1[0], rtol=0, atol=0)
    assert lam_2_total < base_lam_2[0]
    np.testing.assert_allclose(
        lam_2_total,
        base_lam_2[0] + trained_s2.epsilon * trained_s2.get_param("lambda_2")[0],
        rtol=1e-12,
    )


def test_stage2_burgers_residual_linearizes_nonlinear_flux():
    """Stage-2 residual keeps the Burgers cross term once and drops O(eps^2)."""
    key = jax.random.PRNGKey(0)
    lam_1 = 1.3
    lam_2 = 0.4
    s1 = Stage1(
        lb,
        ub,
        in_size=2,
        out_size=1,
        width_size=4,
        depth=2,
        params={
            "lambda_1": jnp.array([lam_1]),
            "log_lambda_2": jnp.log(jnp.array([lam_2])),
        },
        key=key,
    )
    model = Stage2(s1, epsilon=0.07, kappa=jnp.array([2.0, 3.0]), key=key)
    t = jnp.array([0.1, 0.4, 0.8])
    x = jnp.array([-0.7, 0.2, 0.9])

    def full_u(model, ti, xi):
        return model(ti, xi)

    full_t = grad(full_u, argnums=1)
    full_x = grad(full_u, argnums=2)
    full_xx = grad(grad(full_u, argnums=2), argnums=2)

    @partial(vmap, in_axes=(None, 0, 0))
    def full_burgers(model, ti, xi):
        u = model(ti, xi)
        return (
            full_t(model, ti, xi)
            + lam_1 * u * full_x(model, ti, xi)
            - lam_2 * full_xx(model, ti, xi)
        )

    def corr_u(model, ti, xi):
        return model.compute_s2(ti, xi)

    corr_x = vmap(grad(corr_u, argnums=2), in_axes=(None, 0, 0))(model, t, x)
    corr = vmap(corr_u, in_axes=(None, 0, 0))(model, t, x)
    _, staged = residual_s2_burgers(model, t, x)
    expected = full_burgers(model, t, x) - model.epsilon**2 * lam_1 * corr * corr_x
    np.testing.assert_allclose(staged, expected, rtol=1e-12, atol=1e-12)


def beta_burgers(model, t, x):
    """Coefficients of u_t, u, u_x, and u_xx in the linearized operator."""
    lam_1, lam_2 = total_lambdas(model)
    ones = jnp.ones_like(t)
    u = vmap(model)(t, x)

    def u_at(ti, xi):
        return model(ti, xi)

    ux = vmap(grad(u_at, argnums=1))(t, x)
    return jnp.stack([ones, lam_1 * ux, lam_1 * u, lam_2 * ones], axis=-1)


def test_linear_burgers_exact_solution_residual():
    """The synthetic forward-problem data should satisfy the stated PDE."""
    lambda_2 = 0.5 / jnp.pi
    t = jnp.linspace(0.0, 1.0, 9)
    x = jnp.linspace(-1.0, 1.0, 9)

    @vmap
    def exact(ti, xi):
        return jnp.sin(jnp.pi * xi) * jnp.exp(-lambda_2 * jnp.pi**2 * ti)

    u = exact(t, x)
    ut = -lambda_2 * jnp.pi**2 * u
    uxx = -jnp.pi**2 * u
    np.testing.assert_allclose(ut - lambda_2 * uxx, 0.0, atol=1e-14)


def test_scalar_and_unreduced_losses_match_with_collocation_points():
    """Trust-region and scalar optimizers should see the same objective."""
    key = jax.random.PRNGKey(0)
    model = Stage1(
        lb,
        ub,
        in_size=2,
        out_size=1,
        width_size=4,
        depth=2,
        params={
            "lambda_1": jnp.array([0.0]),
            "log_lambda_2": jnp.log(jnp.array([0.5 / jnp.pi])),
        },
        key=key,
    )
    t_data = jnp.array([0.0, 0.3, 0.8])
    x_data = jnp.array([-1.0, -0.1, 0.7])
    u_data = jnp.array([0.0, 0.2, -0.4])
    t_col = jnp.array([0.2, 0.9])
    x_col = jnp.array([-0.6, 0.4])
    gamma = 0.3

    scalar = loss_s1_burgers(model, t_data, x_data, u_data, t_col, x_col, gamma)
    unreduced = loss_s1_burgers_unreduced(
        model, t_data, x_data, u_data, t_col, x_col, gamma
    )
    np.testing.assert_allclose(scalar, jnp.sum(unreduced**2), rtol=1e-12)

    stage2 = Stage2(model, epsilon=0.01, kappa=jnp.array([2.0, 3.0]), key=key)
    scalar = loss_s2_burgers(stage2, t_data, x_data, u_data, t_col, x_col, gamma)
    unreduced = loss_s2_burgers_unreduced(
        stage2, t_data, x_data, u_data, t_col, x_col, gamma
    )
    np.testing.assert_allclose(scalar, jnp.sum(unreduced**2), rtol=1e-12)


def loss_s2_burgers(model, t_data, x_data, u_data, t_col=None, x_col=None, gamma=0.5):
    """Calculates the total loss for the Stage 2 model.

    Parameters
    ----------
    model : Stage2
        The Stage 2 PINN model.
    t_data : jax.Array
        Batch of time coordinates.
    x_data : jax.Array
        Batch of spatial coordinates.
    u_data : jax.Array
        Batch of noisy training data for u.
    t_col : jax.Array
        Time samples where only PDE residual is computed.
    x_col : jax.Array
        X samples where only PDE residual is computed.

    Returns
    -------
    loss : float
        The total loss.

    """
    u_pred, f_pred = residual_s2_burgers(model, t_data, x_data)
    loss_data = squared_error(u_pred, u_data).mean()
    loss_resid = f_pred.ravel()

    if t_col is not None and x_col is not None:
        _, col_resid = residual_s2_burgers(model, t_col, x_col)
        loss_resid = jnp.concatenate((loss_resid, col_resid.ravel()))

    return (1 - gamma) * loss_data + gamma * squared_error(loss_resid).mean()


def loss_s2_burgers_unreduced(
    model, t_data, x_data, u_data, t_col=None, x_col=None, gamma=0.5
):
    """Returns loss residual vector for use with least squares optimizers.

    Parameters
    ----------
    model : Stage2
        The Stage 2 PINN model.
    t_data : jax.Array
        Batch of time coordinates.
    x_data : jax.Array
        Batch of spatial coordinates.
    u_data : jax.Array
        Batch of noisy training data for u.
    t_col : jax.Array
        Time samples where only PDE residual is computed.
    x_col : jax.Array
        X samples where only PDE residual is computed.

    Returns
    -------
    jax.Array
        The total loss (data_loss + residual_loss).

    """
    u_pred, f_pred = residual_s2_burgers(model, t_data, x_data)
    loss_data = jnp.ravel(u_pred - u_data)
    loss_resid = f_pred.ravel()

    if t_col is not None and x_col is not None:
        _, col_resid = residual_s2_burgers(model, t_col, x_col)
        col_resid = col_resid.ravel()
        loss_resid = jnp.concatenate((loss_resid, col_resid))

    loss_data *= jnp.sqrt((1 - gamma) / loss_data.size)
    loss_resid *= jnp.sqrt(gamma / loss_resid.size)

    return jnp.concatenate((loss_data, loss_resid))


def benchmark_state(
    net, stage, name, x_test, u_data_test, LAMBDA_1, LAMBDA_2, step=None
):
    lam_1, lam_2 = total_lambdas(net)
    if step is None:
        print(f"------- Stage {stage} --------")
    else:
        print(f"------- Stage {stage}, step {step} --------")
    print(f"Final params: lambda_1={lam_1:.8f}, ")
    print(f"              lambda_2={lam_2:.8f}")
    print(f"True params:  lambda_1={LAMBDA_1:.8f}, ")
    print(f"              lambda_2={LAMBDA_2:.8f}")
    print(f"lambda_1 error = {jnp.abs(LAMBDA_1 - lam_1):.8e}")
    print(f"lambda_2 error = {jnp.abs(LAMBDA_2 - lam_2):.8e}\n")

    print_errors(net, x_test, u_data_test, check_dim=1)

    if step is not None:
        fname_base = f"plots/{name}_stage_{stage}_step_{step:06d}"
    else:
        fname_base = f"plots/{name}_stage_{stage}_final"

    # plot_2d_solution(
    #     net,
    #     u_true_fn=u_true,  # noqa: E800
    #     figname=f"{fname_base}_2d_grid.png",  # noqa: E800
    # )  # noqa: E800
    plot_2d_residual(
        net,
        residual_fun=residual_s2_burgers if hasattr(net, "s1") else residual_s1_burgers,
        figname=f"{fname_base}_residual_2d.png",
    )
    plt.close("all")


def run_burgers(
    u_true,
    LAMBDA_1,
    LAMBDA_2,
    num_points=20000,
    split_ratio=0.5,
    estimate_params=False,
    n_stages=2,
    optimizer=optax.lbfgs,
    steps=20000,
    lbfgs_steps=1000,
    learning_rate=None,
    adaptive_sample_freq=1000,
    chebyshev=False,
    name="burgers/lbfgs/10k_pts_20k_stps_3layers",
    normal_sample=False,
    normal_sample_scale=0.1,
    **net_kwargs,
):
    """Test Burgers inverse problem optimization.

    Parameters
    ----------
    num_points : int
        Number of training and testing points total.
    split_ratio : float
        Ratio of ``num_points`` to allocate to training.
        Default is 0.5.
    estimate_params : bool
        Whether to estimate the parameters of the Burger's equation lambda1,2.
        Default is False.
    n_stages : int
        Number of stages.
    optimizer : optax.GradientTransformation
        The optimizer to use (e.g., ``optax.lbfgs``).
        Also accepts the input ``"trust region"`` which will perform a trust
        region based optimizer after 1000 iterations of LBFGS.
    steps : int
        Number of optimization steps.
    lbfgs_steps : int
        Number of lbfgs warmup steps.
        Ignored unless trust region method is used.
    learning_rate : float, optional
        Learning rate for the optimizer.
    adaptive_sample_freq : int
        Frequency of adaptive sampling during training. Default is 1000.
    chebyshev : bool
        Whether to use Chebyshev feature mapping instead of Fourier.
    name : str
        Name for the run, used for saving files and benchmarks.
    normal_sample : bool
        Whether to sample additional points from normal distribution around x = 0.
    normal_sample_scale : float
        Standard deviation for additional points around x = 0.
    net_kwargs : dict
        Network hyperparameters for initial Stage1 network.

    """
    in_size = 2
    net_kwargs["lb"] = lb
    net_kwargs["ub"] = ub
    net_kwargs["in_size"] = in_size
    net_kwargs["out_size"] = 1
    net_kwargs.setdefault("width_size", 20)
    net_kwargs.setdefault("depth", 3)
    net_kwargs["params_are_trainable"] = estimate_params

    # stage 1 training data
    x, u_data, key = generate_data(num_points, lb, ub, in_size, u_true)
    if normal_sample:
        x_conc, u_data_conc, key = generate_concentrated_data(
            num_points // 5,
            lb,
            ub,
            in_size,
            u_true,
            axis=1,
            center=0.0,
            scale=normal_sample_scale,
            key=key,
        )
        x = jnp.concatenate([jnp.stack(x), x_conc], axis=1)
        u_data = jnp.concatenate([u_data, u_data_conc])

    # stage 2 training data (if we use same data for stage 2,
    # then we don't gain anything since network already has low loss there)
    x_train_stage2, u_data_train_stage2, key = generate_data(
        num_points // 2, lb, ub, in_size, u_true, key=key
    )
    if normal_sample:
        x_train_conc_stage2, u_data_conc_train_stage2, key = generate_concentrated_data(
            num_points // 5,
            lb,
            ub,
            in_size,
            u_true,
            axis=1,
            center=0.0,
            scale=normal_sample_scale,
            key=key,
        )
        x_train_stage2 = jnp.concatenate(
            [jnp.stack(x_train_stage2), x_train_conc_stage2], axis=1
        )
        u_data_train_stage2 = jnp.concatenate(
            [u_data_train_stage2, u_data_conc_train_stage2]
        )

    split_key, model_key = jax.random.split(key)
    x_train, u_data_train, x_test, u_data_test = train_test_split(
        x, u_data, split_key, split_ratio
    )

    key, lam1_key, lam2_key = jax.random.split(model_key, 3)
    if estimate_params:
        params = {
            "lambda_1": 0.0 + jax.random.normal(lam1_key, (1,)) * 0.1,
            "log_lambda_2": -6.0 + jax.random.normal(lam2_key, (1,)) * 0.1,
        }
    else:
        params = {
            "lambda_1": jnp.array([LAMBDA_1]),
            "log_lambda_2": jnp.log(jnp.array([LAMBDA_2])),
        }
    net = Stage1(**net_kwargs, params=params, key=key)
    num_params = count_params(net)
    print(f"\nNumber of parameters: {num_params}")
    print(f"Number of points:     {u_data_train.shape[0]}\n")
    assert num_params <= (u_data_train.shape[0] / 2), num_params - (
        u_data_train.shape[0] / 2
    )

    adaptive_sample_kwargs = {
        "center": 0.0,
        "scale": normal_sample_scale,
        "normal_sample": (False, normal_sample),
    }
    if optimizer == "trust region":
        net = multistage_trust_region_train(
            net,
            residual_s1_burgers,
            residual_s2_burgers,
            loss_s1_burgers,
            loss_s2_burgers,
            loss_s1_burgers_unreduced,
            loss_s2_burgers_unreduced,
            x_train,
            u_data_train,
            steps=steps,
            lbfgs_steps=lbfgs_steps,
            learning_rate=learning_rate,
            adaptive_sample_freq=adaptive_sample_freq,
            n_stages=n_stages,
            order=((1, 0), (0, 0), (0, 1), (0, 2)),
            beta_fun=beta_burgers,
            chebyshev=chebyshev,
            x_stage2=x_train_stage2,
            training_samples_stage2=u_data_train_stage2,
            net_kwargs_for_save=net_kwargs,
            name=name,
            benchmark_state=partial(
                benchmark_state,
                x_test=x_test,
                u_data_test=u_data_test,
                LAMBDA_1=LAMBDA_1,
                LAMBDA_2=LAMBDA_2,
            ),
            **adaptive_sample_kwargs,
        )
    else:
        net, loss_history = multistage_train(
            net,
            residual_s1_burgers,
            residual_s2_burgers,
            loss_s1_burgers,
            loss_s2_burgers,
            x_train,
            u_data_train,
            optimizer=optimizer,
            steps=steps,
            learning_rate=learning_rate,
            adaptive_sample_freq=adaptive_sample_freq,
            n_stages=n_stages,
            order=((1, 0), (0, 0), (0, 1), (0, 2)),
            beta_fun=beta_burgers,
            chebyshev=chebyshev,
            x_stage2=x_train_stage2,
            training_samples_stage2=u_data_train_stage2,
            net_kwargs_for_save=net_kwargs,
            name=name,
            benchmark_state=partial(
                benchmark_state,
                x_test=x_test,
                u_data_test=u_data_test,
                LAMBDA_1=LAMBDA_1,
                LAMBDA_2=LAMBDA_2,
            ),
            **adaptive_sample_kwargs,
        )

    return net


def main(estimate_params=True, chebyshev=False, nonlinear=False):
    if nonlinear:

        LAMBDA_1 = 1.0
        LAMBDA_2 = 1e-2 / jnp.pi
        _xi, _w = hermgauss(100)

        def u_true(t, x, nu=LAMBDA_2, deg=100):
            """Burger's equation solution using Cole-Hopf transformation.

            The coefficient of the nonlinear term is assumed to be 1.

            Parameters
            ----------
            x : jax.Array
                The spatial coordinates |x|<=1 at which to evaluate the solution.
            t : jax.Array
                The time 0 <= t <= 1 at which to evaluate the solution.
            nu : float
                The viscosity coefficient.
            deg : int
                Number of quadrature points for the Gauss-Hermite integration.

            Returns
            -------
            u : jax.Array
                u(x, t).

            """
            t, x = jnp.atleast_1d(t, x)
            assert t.ndim == x.ndim == 1
            t = t[:, None]
            x = x[:, None]

            xi, w = (_xi, _w) if (deg == 100) else hermgauss(deg)
            eta = jnp.sqrt(4 * nu * t) * xi
            y = jnp.pi * (x - eta)
            f = jnp.exp(-jnp.cos(y) / (2 * jnp.pi * nu))
            return jnp.squeeze(-jnp.dot(jnp.sin(y) * f, w) / jnp.dot(f, w))

    else:
        # For lambda_1 = 0, a true solution is
        # u_true(t, x) = sin(pi*x) * exp(- lambda_2 * pi^2 * t)
        # u_t = lambda_2 * u_xx  # noqa: E800
        # PDE: u_t + lambda_1*u*u_x - lambda_2*u_xx = 0  # noqa: E800
        LAMBDA_1 = 0.0
        LAMBDA_2 = 0.5 / jnp.pi

        @vmap
        def u_true(t, x):
            """Assumes lambda_1 = 0."""
            return jnp.sin(jnp.pi * x) * jnp.exp(-LAMBDA_2 * jnp.pi**2 * t)

    name_start = "nonlinear/" if nonlinear else ""
    name_start += "estimate/" if estimate_params else ""
    name_start += "chebyshev/" if chebyshev else "fourier/"

    print()
    print("--------------------- Run 1 ---------------------")
    name = "burgers/lbfgs/" + name_start + "10k_pts_20k_stps_3_width_30_layers"
    run_burgers(
        u_true,
        LAMBDA_1,
        LAMBDA_2,
        num_points=20000,
        split_ratio=0.5,
        estimate_params=estimate_params,
        n_stages=2,
        optimizer=optax.lbfgs,
        steps=20000,
        adaptive_sample_freq=500,
        chebyshev=chebyshev,
        name=name,
        normal_sample=nonlinear,
        width_size=30,
    )
    plot_loss(
        [f"checkpoints/{name}_stage_0/", f"checkpoints/{name}_stage_1/"],
        figname="plots/" + name + "_loss.pdf",
        title="Loss, sample size 10k, model size small",
    )

    print("--------------------- Run 2 ---------------------")
    name = "burgers/lbfgs/" + name_start + "15k_pts_20k_stps_4layers"
    run_burgers(
        u_true,
        LAMBDA_1,
        LAMBDA_2,
        num_points=30000,
        split_ratio=0.5,
        estimate_params=estimate_params,
        n_stages=2,
        optimizer=optax.lbfgs,
        steps=20000,
        chebyshev=chebyshev,
        name=name,
        normal_sample=nonlinear,
        depth=4,
    )
    plot_loss(
        [f"checkpoints/{name}_stage_0/", f"checkpoints/{name}_stage_1/"],
        figname="plots/" + name + "_loss.pdf",
        title="Loss, sample size 15k, model size medium",
    )

    print("--------------------- Run 3 ---------------------")
    name = "burgers/trust_region/" + name_start + "5k_pts_100_stps_3_width_30_layers"
    run_burgers(
        u_true,
        LAMBDA_1,
        LAMBDA_2,
        num_points=10000,
        split_ratio=0.5,
        estimate_params=estimate_params,
        n_stages=2,
        optimizer="trust region",
        steps=100,
        lbfgs_steps=2000,
        adaptive_sample_freq=20,
        chebyshev=chebyshev,
        name=name,
        normal_sample=nonlinear,
        depth=2,
        width_size=30,
    )


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")

    # python -u tests/test_burgers.py > tests/log/test_burgers.log 2>&1
    main(estimate_params=False, nonlinear=False)
    main(estimate_params=True, nonlinear=False)
    main(estimate_params=False, nonlinear=True)
    main(estimate_params=True, nonlinear=True)
    # main(estimate_params=False, chebyshev=True)  # noqa: E800
    # main(estimate_params=True, chebyshev=True)  # noqa: E800
