"""Test saving and loading is serialized properly for multi-stage network."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from paramax import unwrap

from multistage import Stage1, Stage2, load, save
from multistage._multistage import (
    _feature_scale_from_frequency,
    _train,
    _trainable_params_or_none,
)


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


def test_stages_without_params_use_defaults():
    """Models without inverse parameters should still be valid pytrees."""
    key = jax.random.PRNGKey(42)
    lb = jnp.array([0.0])
    ub = jnp.array([1.0])
    s1 = Stage1(lb, ub, in_size=1, out_size=1, key=key)
    s2 = Stage2(s1, epsilon=0.1, kappa=jnp.array([3.0]), key=key)

    assert tuple(s1.params.items()) == ()
    assert tuple(s2.params.items()) == ()
    assert s1.get_param("missing", 1.5) == 1.5
    assert s2.get_param("missing", 2.5) == 2.5
    eqx.partition(s2, eqx.is_inexact_array)


def test_frozen_params_do_not_seed_stage_corrections():
    """Frozen inverse parameters should not create empty trainable corrections."""
    net, _ = _get_s1(params_are_trainable=False)
    assert _trainable_params_or_none(net._params) is None


def test_stage2_frequency_compensates_for_linear_initialization():
    """``kappa`` is a target frequency, not the raw first-layer scale factor."""
    kappa = jnp.array([4.0, 7.0])
    assert jnp.allclose(_feature_scale_from_frequency(kappa, in_size=2), kappa * 6**0.5)


def test_stage2_feature_map_modes():
    """The default feature map is axis-separable; random mode is dense."""
    net, _ = _get_s1(params_are_trainable=False)
    separable = Stage2(
        net,
        epsilon=0.1,
        kappa=jnp.array([2.0, 3.0]),
        width_size=5,
        feature_map="separable",
    )
    expected = jax.nn.one_hot(jnp.arange(5) % 2, 2)
    np.testing.assert_allclose(unwrap(separable._feature_mask), expected)

    random = Stage2(
        net,
        epsilon=0.1,
        kappa=jnp.array([2.0, 3.0]),
        width_size=5,
        feature_map="random",
    )
    np.testing.assert_allclose(unwrap(random._feature_mask), jnp.ones((5, 2)))


def test_train_uses_initial_adaptive_sample():
    """Adaptive collocation should affect even short training runs."""
    key = jax.random.PRNGKey(0)
    net = Stage1(
        jnp.array([0.0]),
        jnp.array([1.0]),
        in_size=1,
        out_size=1,
        width_size=2,
        depth=1,
        key=key,
    )
    x = [jnp.array([0.2, 0.8])]
    y = jnp.array([0.0, 0.0])
    calls = []

    def adaptive_sampler(net, key=None):
        calls.append(True)
        return [jnp.array([0.5])], key

    def loss_fun(net, x_data, u_data, x_col):
        u_pred = jax.vmap(net)(x_data)
        return jnp.mean((u_pred - u_data) ** 2) + 0.0 * jnp.mean(x_col)

    _train(
        net,
        loss_fun,
        x,
        y,
        optimizer=optax.sgd,
        steps=1,
        learning_rate=0.0,
        adaptive_sampler=adaptive_sampler,
        adaptive_sample_freq=10,
        return_loss_history=False,
        print_every=1,
        checkpoint_path=None,
    )

    assert calls == [True]


def test_train_print_every_zero_only_logs_final():
    """Disabling periodic logs should not break the training loop."""
    key = jax.random.PRNGKey(0)
    net = Stage1(
        jnp.array([0.0]),
        jnp.array([1.0]),
        in_size=1,
        out_size=1,
        width_size=2,
        depth=1,
        key=key,
    )
    x = [jnp.array([0.2, 0.8])]
    y = jnp.array([0.0, 0.0])

    def loss_fun(net, x_data, u_data, x_col):
        del x_col
        u_pred = jax.vmap(net)(x_data)
        return jnp.mean((u_pred - u_data) ** 2)

    _train(
        net,
        loss_fun,
        x,
        y,
        optimizer=optax.sgd,
        steps=2,
        learning_rate=0.0,
        adaptive_sampler=None,
        return_loss_history=False,
        print_every=0,
        checkpoint_path=None,
    )


@pytest.mark.parametrize("params_are_trainable", [True, False])
def test_save_load_stage1(params_are_trainable, tmp_path):
    """Test saving and loading is serialized properly for single stage network."""
    net, kwargs = _get_s1(params_are_trainable)
    filename = tmp_path / "test_save_s1.eqx"
    save(filename, net, **kwargs)
    loaded_net = load(filename, Stage1)
    assert eqx.tree_equal(loaded_net, net)
    assert loaded_net == net


@pytest.mark.parametrize("params_are_trainable", [True, False])
def test_save_load_stage2(params_are_trainable, tmp_path):
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
    filename = tmp_path / "test_save_s2.eqx"
    save(filename, net, **kwargs)
    loaded_net = load(filename, Stage2, s1=s1, activation=jnp.cos)

    assert eqx.tree_equal(loaded_net, net)
    assert loaded_net == net
