"""Test saving and loading is serialized properly for multi-stage network."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from multistage import Stage1, Stage2, load, save


def _get_s1(params_are_trainable):
    key = jax.random.PRNGKey(42)
    key, lam1_key, lam2_key = jax.random.split(key, 3)
    params = {
        "lambda_1": 0.0 + jax.random.normal(lam1_key, (1,)) * 0.1,
        "log_lambda_2": -6.0 + jax.random.normal(lam2_key, (1,)) * 0.1,
    }

    lb = jnp.array([0.0, -1.0])
    ub = jnp.array([1.0, 1.0])

    kwargs = dict(
        lb=lb,
        ub=ub,
        in_size=2,
        out_size=1,
        width_size=20,
        depth=4,
        params_are_trainable=params_are_trainable,
    )
    return Stage1(params=params, key=key, **kwargs), kwargs


@pytest.mark.parametrize("params_are_trainable", [True, False])
def test_save_load_stage1(params_are_trainable):
    """Test saving and loading is serialized properly for single stage network."""
    net, kwargs = _get_s1(params_are_trainable)
    save("test_save_s1.eqx", net, **kwargs)
    loaded_net = load("test_save_s1.eqx", Stage1)
    assert eqx.tree_equal(loaded_net, net)
    assert loaded_net == net


@pytest.mark.parametrize("params_are_trainable", [True, False])
def test_save_load_stage2(params_are_trainable):
    """Test saving and loading is serialized properly for multi-stage network."""
    key = jax.random.PRNGKey(43)
    key, lam1_key, lam2_key = jax.random.split(key, 3)
    params = {
        "lambda_1": 0.0 + jax.random.normal(lam1_key, (1,)) * 0.1,
        "log_lambda_2": -6.0 + jax.random.normal(lam2_key, (1,)) * 0.1,
        "new_param": 8.0,
    }

    s1, _ = _get_s1(params_are_trainable)
    kwargs = dict(epsilon=3.0, kappa=jnp.array([3, 4.6]), width_size=8, depth=3)
    net = Stage2(s1, params=params, key=key, activation=jnp.cos, **kwargs)
    save("test_save_s2.eqx", net, **kwargs)
    loaded_net = load("test_save_s2.eqx", Stage2, s1=s1, activation=jnp.cos)

    assert eqx.tree_equal(loaded_net, net)
    assert loaded_net == net
