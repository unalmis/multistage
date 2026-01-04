"""Input, output, saving, etc.."""

import json
import os
from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from paramax import non_trainable, unwrap

from ._utils import is_not_trainable

_sentinel_key = "_is_array_to_repack"


class _ParamContainer(eqx.Module):
    """Class to get JAX working with trainable, frozen, static API."""

    keys: Tuple[str, ...]
    vals: Tuple[jax.Array]

    def __init__(self, params: dict[str, jax.Array]):
        if params is None:
            params = {}
        self.keys = tuple(params.keys())
        self.vals = tuple(params.values())

    def items(self):
        return zip(self.keys, self.vals)

    def __getitem__(self, key):
        try:
            idx = self.keys.index(key)
            return self.vals[idx]
        except ValueError:
            raise KeyError(f"Key '{key}' not found in ParamContainer.")

    def get(self, key, default=None):
        try:
            idx = self.keys.index(key)
            return self.vals[idx]
        except ValueError:
            return default

    def __repr__(self):
        items = [f"'{k}': {v}" for k, v in self.items()]
        return f"ParamContainer({{{', '.join(items)}}})"


def _save_skeleton(obj):
    """Recursively saves the skeleton of a Pytree for serialization."""
    non_trainable_leaf = False
    if is_not_trainable(obj):
        obj = unwrap(obj)
        non_trainable_leaf = True
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        return {
            _sentinel_key: True,
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "non_trainable_leaf": non_trainable_leaf,
        }
    if isinstance(obj, dict):
        return {key: _save_skeleton(val) for key, val in obj.items()}
    return obj


def _load_skeleton(obj):
    """Recursively loads the skeleton of a Pytree for serialization."""
    if isinstance(obj, dict):
        if obj.get(_sentinel_key):
            out = jnp.zeros(obj["shape"], dtype=obj["dtype"])
            if obj.get("non_trainable_leaf"):
                out = non_trainable(out)
            return out
        return {key: _load_skeleton(val) for key, val in obj.items()}
    return obj


def save(filename, model, **kwargs):
    """Save the model weights and configuration to a single file.

    This method separates the configuration (JSON header) from the
    weights (binary payload). JAX arrays in the configuration are
    converted to metadata schemas to preserve type and shape information
    without writing large binary data to the text header.

    Parameters
    ----------
    filename : str
        Path to save the file (usually ending in .eqx).
    model : eqx.Module
        Model to save.
    kwargs : dict
        Serializable things that were given to the constructor to make this model.
        Excluding ``params`` to infer for inverse method.
        If an input was another Neural network, it is likely best to save this
        separately and pass it in as an additional kwarg to load.

    """
    skeleton = {k: _save_skeleton(v) for k, v in kwargs.items()}

    assert "_ParamContainer" not in skeleton
    for attribute, value in model.__dict__.items():
        if isinstance(value, _ParamContainer):
            container = skeleton.setdefault("_ParamContainer", {})
            if attribute[0] == "_":
                attribute = attribute[1:]
            container[attribute] = {k: _save_skeleton(v) for k, v in value.items()}

    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    with open(filename, "wb") as f:
        # Write skeleton to rebuild Pytree.
        f.write((json.dumps(skeleton) + "\n").encode())
        # Write binary data to fill skeleton.
        eqx.tree_serialise_leaves(f, model)


def load(filename, model_constructor, **kwargs):
    """Load the model from a file containing Pytree skeleton and binary data.

    Parameters
    ----------
    filename : str
        File to load from.
    model_constructor : callable
        Function to build model.
    kwargs : dict
        Additional kwargs to pass to ``model_constructor``, typically because
        these kwargs could not be serialized.

    """
    with open(filename, "rb") as f:
        config = json.loads(f.readline().decode())

        skeleton = {k: _load_skeleton(v) for k, v in config.items()}
        container = skeleton.pop("_ParamContainer", {})
        for k, v in container.items():
            skeleton[k] = v

        skeleton = model_constructor(**skeleton, **kwargs)
        return eqx.tree_deserialise_leaves(f, skeleton)


def checkpoint_manager(checkpoint_path, max_to_keep=3, create=True):
    """Get thing to make checkpoint for optimizer model.

    Parameters
    ----------
    checkpoint_path : str
        Directory in current path to save file.
    max_to_keep : int
        Maximum number of checkpoints to retain in directory.

    Returns
    -------
    manager : CheckpointManager

    """
    options = ocp.CheckpointManagerOptions(max_to_keep=max_to_keep, create=create)
    return ocp.CheckpointManager(os.path.abspath(checkpoint_path), options=options)
