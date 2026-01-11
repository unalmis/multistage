"""Test multi-stage network on the the 2D Burger's equation inverse problem."""

from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
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
    loss_resid = squared_error(f_pred).mean()

    if t_col is not None and x_col is not None:
        _, col_resid = residual_s1_burgers(model, t_col, x_col)
        col_resid = squared_error(col_resid).mean()
    else:
        col_resid = 0

    return (1 - gamma) * loss_data + gamma * (loss_resid + col_resid)


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


u_fn_s2 = Stage2.__call__
u_t_s2 = grad(u_fn_s2, argnums=1)
u_x_s2 = grad(u_fn_s2, argnums=2)
u_xx_s2 = grad(grad(u_fn_s2, argnums=2), argnums=2)


@partial(grad, argnums=2)
def _u_prod_x(model, t, x):
    # valid for stage 2; trivial to add other combinations for stage 2,
    # but those are of order epsilon^2 anyway.
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
    ut = u_t_s2(model, t, x)
    ux = u_x_s2(model, t, x)
    uxx = u_xx_s2(model, t, x)
    lam_1, lam_2 = total_lambdas(model)
    f = ut + lam_1 * u * ux - lam_2 * uxx
    f = f + model.epsilon * lam_1 * _u_prod_x(model, t, x)
    return u, f


def total_lambdas(model):

    epsilons = [model.epsilon]
    lams_1 = [model.get_param("lambda_1", 0.0)]
    lams_2 = [jnp.exp(model.get_param("log_lambda_2", -jnp.inf))]
    s1 = model
    while hasattr(s1, "s1"):
        s1 = s1.s1
        lams_1.append(s1.get_param("lambda_1", 0.0))
        lams_2.append(jnp.exp(s1.get_param("log_lambda_2", -jnp.inf)))
        epsilons.append(s1.epsilon)

    lam_1 = 0.0
    lam_2 = 0.0
    for i in range(len(epsilons)):
        lam_1 += epsilons[i] * lams_1[i]
        lam_2 += epsilons[i] * lams_2[i]

    if hasattr(lam_1, "size"):
        lam_1 = lam_1[0]
    if hasattr(lam_2, "size"):
        lam_2 = lam_2[0]
    return lam_1, lam_2


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
    loss_resid = squared_error(f_pred).mean()

    if t_col is not None and x_col is not None:
        _, col_resid = residual_s2_burgers(model, t_col, x_col)
        col_resid = squared_error(col_resid).mean()
    else:
        col_resid = 0

    return (1 - gamma) * loss_data + gamma * (loss_resid + col_resid)


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
            scale=LAMBDA_1,
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
            scale=LAMBDA_1,
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
        "scale": LAMBDA_1,
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
            order=(1, 2),
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
            order=(1, 2),
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
