"""multistage: Multistage neural networks with JAX."""

from . import _version
from ._io_utils import load, save
from ._multistage import (
    MultiCorrectionStage,
    Stage1,
    Stage2,
    multistage_train,
    multistage_trust_region_train,
    select_gamma,
)
from ._plot import plot_2d_residual, plot_2d_solution, plot_loss

__all__ = [
    "Stage1",
    "Stage2",
    "MultiCorrectionStage",
    "multistage_train",
    "multistage_trust_region_train",
    "select_gamma",
    "plot_2d_residual",
    "plot_2d_solution",
    "plot_loss",
    "load",
    "save",
]

__version__ = _version.get_versions()["version"]
