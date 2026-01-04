"""Plotting utilities."""

import logging
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import orbax.checkpoint as ocp
from jax import vmap
from matplotlib.colors import LogNorm

from ._io_utils import checkpoint_manager


def plot_2d_solution(net, u_true_fn=None, resolution=256, figname="2d_solution.pdf"):
    """Plots solution, truth, and error on a uniform 2D grid.

    Parameters
    ----------
    net : eqx.Module
        The neural network model.
    u_true_fn : callable
        Function that returns the true value of the solution.
    resolution : int
        Grid resolution for the plot.
    figname : str
        Filename to save the plot.

    Returns
    -------
    fig, ax
        Matplotlib fig and ax.

    """
    figname = Path(figname)
    figname.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams["figure.constrained_layout.use"] = True
    lb, ub = net.lb, net.ub

    t = jnp.linspace(lb[0], ub[0], resolution)
    x = jnp.linspace(lb[1], ub[1], resolution)
    T, X = jnp.meshgrid(t, x)
    t, x = map(jnp.ravel, (T, X))

    U_pred = vmap(net)(t, x).reshape(T.shape)

    if u_true_fn is None:
        fig, ax = plt.subplots(figsize=(6, 5))
        cp = ax.pcolormesh(T, X, U_pred, cmap="viridis", shading="auto")
        fig.colorbar(cp, ax=ax, label="u(t, x)")
        ax.set_xlabel("t")
        ax.set_ylabel("x")
        ax.set_title("Predicted Solution")
        fig.savefig(figname)
        plt.close(fig)
        return fig, ax

    U_true = u_true_fn(t, x).reshape(T.shape)
    err = jnp.abs(U_true - U_pred)
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    c1 = ax[0].pcolormesh(T, X, U_pred, cmap="viridis", shading="auto")
    ax[0].set_title("Prediction")
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("x")
    fig.colorbar(c1, ax=ax[0])

    c2 = ax[1].pcolormesh(T, X, U_true, cmap="viridis", shading="auto")
    ax[1].set_title("Truth")
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("x")
    fig.colorbar(c2, ax=ax[1])

    eps = jnp.finfo(jnp.array(1.0).dtype).eps
    c3 = ax[2].pcolormesh(T, X, err + eps, cmap="turbo", norm=LogNorm(), shading="auto")
    ax[2].set_title(
        f"Abs Err. (Max: {err.max():.2e}, Mean: {err.mean():.2e}, Min: {err.min():.2e})"
    )
    ax[2].set_xlabel("t")
    ax[2].set_ylabel("x")
    fig.colorbar(c3, ax=ax[2])
    fig.savefig(figname)
    plt.close(fig)

    return fig, ax


def plot_2d_residual(
    net, residual_fun, resolution=256, figname="2d_residual.pdf", cmap="seismic"
):
    """Plots the signed PDE residual on a uniform 2D grid with a linear scale.

    Parameters
    ----------
    net : eqx.Module
        The neural network model.
    residual_fun : callable
        Function that returns (u, f), where f is the residual.
    resolution : int
        Grid resolution for the plot.
    figname : str
        Filename to save the plot.
    cmap : str
        Matplotlib colormap.

    Returns
    -------
    fig, ax
        Matplotlib fig, ax.

    """
    figname = Path(figname)
    figname.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams["figure.constrained_layout.use"] = True
    lb, ub = net.lb, net.ub
    t = jnp.linspace(lb[0], ub[0], resolution)
    x = jnp.linspace(lb[1], ub[1], resolution)
    T, X = jnp.meshgrid(t, x)
    t_flat, x_flat = map(jnp.ravel, (T, X))

    _, f_pred = residual_fun(net, t_flat, x_flat)
    F_grid = f_pred.reshape(T.shape)

    fig, ax = plt.subplots(figsize=(7, 6))
    limit = jnp.max(jnp.abs(F_grid))
    cp = ax.pcolormesh(T, X, F_grid, cmap=cmap, vmin=-limit, vmax=limit, shading="auto")
    cbar = fig.colorbar(cp, ax=ax)
    cbar.set_label("Residual value")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_title(
        f"Signed PDE Residual\n"
        f"Min: {F_grid.min():.2e}, Max: {F_grid.max():.2e}, "
        f"RMSE: {jnp.sqrt(jnp.mean(F_grid**2)):.2e}"
    )
    fig.savefig(figname)
    plt.close(fig)

    return fig, ax


def plot_loss(checkpoint_path, figname, title="Loss"):
    """Plot loss history from specified path.

    Parameters
    ----------
    checkpoint_path : str or list[str]
        Path to the directory containing checkpoints.
    figname : str
        Filename to save the plot.
    title : str
        Title for plot.

    Returns
    -------
    fig, ax
        Matplotlib fig, ax.

    """
    figname = Path(figname)
    figname.parent.mkdir(parents=True, exist_ok=True)
    ocp_args = ocp.args.StandardRestore()

    logger = logging.getLogger("absl")
    level = logger.level

    plt.rcParams["figure.constrained_layout.use"] = True
    fig, ax = plt.subplots()

    if isinstance(checkpoint_path, str):
        checkpoint_path = [checkpoint_path]

    loss = np.array([])
    for i, path in enumerate(checkpoint_path):
        manager = checkpoint_manager(path, create=False)
        step = manager.latest_step()
        try:
            logger.setLevel(logging.ERROR)
            restored = manager.restore(step, args=ocp_args)
        finally:
            logger.setLevel(level)
        restored = np.asarray(restored["loss_history"])
        loss = np.concatenate((loss, restored))

        min_line_y = 0.9 * restored.min()
        coeff, expo = f"{min_line_y:.1e}".split("e")
        ax.axhline(
            y=min_line_y,
            color="k",
            linestyle="--",
            linewidth=1.5,
            label=rf"${coeff} \times 10^{{{int(expo)}}}$",
        )

    ax.plot(loss, label="Training Loss", linewidth=2)
    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    fig.savefig(figname)
    plt.close(fig)

    return fig, ax
