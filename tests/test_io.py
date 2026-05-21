"""Test saving and loading is serialized properly for multi-stage network."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from paramax import unwrap

import multistage._multistage as multistage_module
import multistage._plot as plot_module
from multistage import Stage1, Stage2, load, plot_loss, save
from multistage._multistage import (
    _feature_scale_from_frequency,
    _stage_correction_params_or_none,
    _train,
    _trainable_params_or_none,
    _trust_region_train,
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
    assert _stage_correction_params_or_none(net._params) is None


def test_next_stage_inverse_params_are_zero_signed_corrections():
    """New inverse stages train corrections, not shifted copies of totals."""
    net, _ = _get_s1(params_are_trainable=True)
    corrections = _stage_correction_params_or_none(net._params)

    assert corrections.keys == ("lambda_1", "lambda_2")
    np.testing.assert_allclose(corrections["lambda_1"], jnp.array([0.0]))
    np.testing.assert_allclose(corrections["lambda_2"], jnp.array([0.0]))


def test_stage_correction_requires_explicit_unknown_transform_map():
    """Unknown transformed params should not get an invented neutral value."""
    net = Stage1(
        jnp.array([0.0]),
        jnp.array([1.0]),
        in_size=1,
        out_size=1,
        width_size=2,
        depth=1,
        params={"log_alpha": jnp.array([0.3])},
        params_are_trainable=True,
    )

    with pytest.raises(ValueError, match="log_alpha"):
        _stage_correction_params_or_none(net._params)

    corrections = _stage_correction_params_or_none(
        net._params, correction_param_map={"log_alpha": "alpha"}
    )
    assert corrections.keys == ("alpha",)
    np.testing.assert_allclose(corrections["alpha"], jnp.array([0.0]))


def test_stage_correction_rejects_parameter_map_collisions():
    """Two current params cannot silently initialize one correction param."""
    net = Stage1(
        jnp.array([0.0]),
        jnp.array([1.0]),
        in_size=1,
        out_size=1,
        width_size=2,
        depth=1,
        params={
            "lambda_2": jnp.array([0.1]),
            "log_lambda_2": jnp.log(jnp.array([0.2])),
        },
        params_are_trainable=True,
    )

    with pytest.raises(ValueError, match="lambda_2"):
        _stage_correction_params_or_none(net._params)


def test_stage2_get_param_is_stage_local():
    """Correction-stage params are local; PDE code combines stages explicitly."""
    s1, _ = _get_s1(params_are_trainable=False)
    s2 = Stage2(
        s1,
        epsilon=0.1,
        kappa=jnp.array([2.0, 3.0]),
        params={"lambda_1": jnp.array([3.0])},
        params_are_trainable=True,
    )

    np.testing.assert_allclose(s2.get_param("lambda_1"), jnp.array([3.0]))
    assert s2.get_param("log_lambda_2", None) is None
    assert s2.get_param("missing", 4.0) == 4.0


def test_stage2_allows_transformed_correction_params():
    """Manual later-stage params may use transformed optimization variables."""
    s1, _ = _get_s1(params_are_trainable=False)
    s2 = Stage2(
        s1,
        epsilon=0.1,
        kappa=jnp.array([2.0, 3.0]),
        params={"log_lambda_2": jnp.log(jnp.array([0.5]))},
        params_are_trainable=True,
    )

    np.testing.assert_allclose(s2.get_param("log_lambda_2"), jnp.log(jnp.array([0.5])))


def test_stage2_training_does_not_modify_trained_stage1():
    """Stage 2 optimization must keep the trained previous stage frozen."""
    key = jax.random.PRNGKey(0)
    s1_key, s2_key = jax.random.split(key)
    s1 = Stage1(
        jnp.array([0.0]),
        jnp.array([1.0]),
        in_size=1,
        out_size=1,
        width_size=3,
        depth=2,
        params={
            "lambda_1": jnp.array([0.3]),
            "log_lambda_2": jnp.log(jnp.array([0.2])),
        },
        params_are_trainable=True,
        key=s1_key,
    )
    x = [jnp.linspace(0.0, 1.0, 8)]
    y_stage1 = jnp.linspace(-0.2, 0.8, 8)
    initial_u = jax.vmap(s1)(*x)
    initial_lambda_1 = s1.get_param("lambda_1")
    initial_log_lambda_2 = s1.get_param("log_lambda_2")

    def stage1_loss_fun(net, x_data, u_data, x_col):
        del x_col
        loss_u = jnp.mean((jax.vmap(net)(x_data) - u_data) ** 2)
        loss_lam_1 = jnp.mean((net.get_param("lambda_1") - 0.7) ** 2)
        loss_lam_2 = jnp.mean((jnp.exp(net.get_param("log_lambda_2")) - 0.4) ** 2)
        return loss_u + loss_lam_1 + loss_lam_2

    trained_s1 = _train(
        s1,
        stage1_loss_fun,
        x,
        y_stage1,
        optimizer=optax.sgd,
        steps=5,
        learning_rate=1e-2,
        adaptive_sampler=None,
        return_loss_history=False,
        print_every=0,
        checkpoint_path=None,
        normalize_loss=False,
    )
    trained_stage1_u = jax.vmap(trained_s1)(*x)

    assert not jnp.allclose(trained_stage1_u, initial_u)
    assert not jnp.allclose(trained_s1.get_param("lambda_1"), initial_lambda_1)
    assert not jnp.allclose(trained_s1.get_param("log_lambda_2"), initial_log_lambda_2)

    s2 = Stage2(
        trained_s1,
        epsilon=0.2,
        kappa=jnp.array([2.0]),
        width_size=4,
        depth=2,
        params={"lambda_1": jnp.array([0.1])},
        params_are_trainable=True,
        key=s2_key,
    )
    y_stage2 = jnp.ones(8)
    initial_correction = jax.vmap(s2.compute_s2)(*x)

    def stage2_loss_fun(net, x_data, u_data, x_col):
        del x_col
        return jnp.mean((jax.vmap(net.compute_s2)(x_data) - u_data) ** 2)

    trained_s2 = _train(
        s2,
        stage2_loss_fun,
        x,
        y_stage2,
        optimizer=optax.sgd,
        steps=3,
        learning_rate=1e-2,
        adaptive_sampler=None,
        return_loss_history=False,
        print_every=0,
        checkpoint_path=None,
        normalize_loss=False,
    )

    assert eqx.tree_equal(trained_s2.s1, unwrap(trained_s1))
    np.testing.assert_allclose(
        jax.vmap(trained_s2.s1)(*x), trained_stage1_u, rtol=0, atol=0
    )
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
    assert not jnp.allclose(jax.vmap(trained_s2.compute_s2)(*x), initial_correction)


def test_stage2_frequency_compensates_for_linear_initialization():
    """``kappa`` is a target frequency, not the raw first-layer scale factor."""
    kappa = jnp.array([4.0, 7.0])
    assert jnp.allclose(_feature_scale_from_frequency(kappa, in_size=2), kappa * 6**0.5)
    assert jnp.allclose(
        _feature_scale_from_frequency(kappa, in_size=2, feature_map="separable"),
        kappa * 6**0.5,
    )
    assert jnp.allclose(
        _feature_scale_from_frequency(kappa, in_size=2, feature_map="random"),
        kappa * 3**0.5,
    )


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


def test_stage2_random_feature_effective_wave_vectors_are_not_overscaled():
    """Dense random features should have the same RMS wave-vector norm as kappa."""
    in_size = 3
    width_size = 8192
    kappa_value = 5.0
    key = jax.random.PRNGKey(123)
    s1_key, separable_key, random_key = jax.random.split(key, 3)
    s1 = Stage1(
        jnp.zeros(in_size),
        jnp.ones(in_size),
        in_size=in_size,
        out_size=1,
        width_size=2,
        depth=1,
        key=s1_key,
    )

    def effective_wave_vector_rms(feature_map, key):
        stage = Stage2(
            s1,
            epsilon=0.1,
            kappa=jnp.full((in_size,), kappa_value),
            width_size=width_size,
            depth=1,
            feature_map=feature_map,
            key=key,
        )
        feature_scale = _feature_scale_from_frequency(
            stage.kappa, stage.in_size, stage._feature_map
        )
        wave_vectors = stage._first.weight * unwrap(stage._feature_mask) * feature_scale
        return jnp.sqrt(jnp.mean(jnp.sum(wave_vectors**2, axis=1)))

    separable_rms = effective_wave_vector_rms("separable", separable_key)
    random_rms = effective_wave_vector_rms("random", random_key)

    np.testing.assert_allclose(separable_rms, kappa_value, rtol=0.04)
    np.testing.assert_allclose(random_rms, kappa_value, rtol=0.04)
    assert random_rms < 1.2 * kappa_value


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


def test_train_checkpoints_loss_ref(monkeypatch):
    """Normalized training checkpoints should preserve their objective scale."""
    saved = []

    class FakeManager:
        def latest_step(self):
            return None

        def save(self, step, args):
            saved.append((step, args))

        def wait_until_finished(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(
        multistage_module, "checkpoint_manager", lambda checkpoint_path: FakeManager()
    )
    monkeypatch.setattr(multistage_module.ocp.args, "StandardSave", lambda args: args)

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
        steps=1,
        learning_rate=0.0,
        adaptive_sampler=None,
        return_loss_history=True,
        print_every=0,
        checkpoint_path="fake",
        checkpoint_every=1,
    )

    assert saved
    assert saved[-1][0] == 0
    assert "loss_ref" in saved[-1][1]
    assert jnp.isfinite(saved[-1][1]["loss_ref"])


def test_train_resume_checkpoints_first_new_step(monkeypatch):
    """A resumed short run should checkpoint the first newly completed step."""
    saved = []

    class FakeManager:
        def latest_step(self):
            return 1

        def restore(self, step, args):
            assert step == 1
            return args

        def save(self, step, args):
            del args
            saved.append(step)

        def wait_until_finished(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(
        multistage_module, "checkpoint_manager", lambda checkpoint_path: FakeManager()
    )
    monkeypatch.setattr(
        multistage_module.ocp.args, "StandardRestore", lambda args: args
    )
    monkeypatch.setattr(multistage_module.ocp.args, "StandardSave", lambda args: args)

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
        steps=3,
        learning_rate=0.0,
        adaptive_sampler=None,
        return_loss_history=False,
        print_every=0,
        checkpoint_path="fake",
        checkpoint_every=1,
    )

    assert saved == [2]


def test_train_checkpoint_every_zero_ignores_checkpoint_path(monkeypatch):
    """Disabling checkpoints should also disable stale checkpoint restores."""

    def fail_checkpoint_manager(checkpoint_path):
        del checkpoint_path
        raise AssertionError("checkpoint manager should not be created")

    monkeypatch.setattr(
        multistage_module, "checkpoint_manager", fail_checkpoint_manager
    )

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
        steps=1,
        learning_rate=0.0,
        adaptive_sampler=None,
        return_loss_history=False,
        print_every=0,
        checkpoint_path="fake",
        checkpoint_every=0,
    )


def test_trust_region_resume_uses_completed_step_count(monkeypatch):
    """Trust-region checkpoints are saved by completed steps, not step index."""
    solve_steps = []

    class FakeManager:
        def latest_step(self):
            return 2

        def restore(self, step, args):
            del args
            assert step == 2
            return {"trainable": initial_trainable, "loss_ref": jnp.array(1.0)}

        def save(self, step, args):
            del step, args

        def wait_until_finished(self):
            pass

        def close(self):
            pass

    class FakeSolution:
        def __init__(self, value):
            self.value = value
            self.result = multistage_module.optimistix.RESULTS.successful

    def fake_least_squares(fn, solver, trainable, args, max_steps, throw):
        del fn, solver, args, throw
        solve_steps.append(max_steps)
        return FakeSolution(trainable)

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
    initial_trainable = multistage_module.partition(net)[0]

    monkeypatch.setattr(
        multistage_module, "checkpoint_manager", lambda checkpoint_path: FakeManager()
    )
    monkeypatch.setattr(
        multistage_module.ocp.args, "StandardRestore", lambda args: args
    )
    monkeypatch.setattr(multistage_module.ocp.args, "StandardSave", lambda args: args)
    monkeypatch.setattr(
        multistage_module.optimistix, "least_squares", fake_least_squares
    )

    def loss_fun_unreduced(net, x_data, u_data, x_col):
        del x_col
        return jax.vmap(net)(x_data) - u_data

    _trust_region_train(
        net,
        loss_fun_unreduced,
        [jnp.array([0.2, 0.8])],
        jnp.array([0.0, 0.0]),
        steps=5,
        rtol=1e-6,
        atol=1e-6,
        linear_solver=multistage_module.lx.QR(),
        adaptive_sampler=None,
        adaptive_sample_freq=0,
        checkpoint_path="fake",
        checkpoint_every=2,
    )

    assert solve_steps == [2, 1]


def test_trust_region_checkpoint_every_zero_ignores_checkpoint_path(monkeypatch):
    """Disabling trust-region checkpoints should also disable restores."""

    def fail_checkpoint_manager(checkpoint_path):
        del checkpoint_path
        raise AssertionError("checkpoint manager should not be created")

    monkeypatch.setattr(
        multistage_module, "checkpoint_manager", fail_checkpoint_manager
    )

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

    def loss_fun_unreduced(net, x_data, u_data, x_col):
        del x_col
        return jax.vmap(net)(x_data) - u_data

    trained = _trust_region_train(
        net,
        loss_fun_unreduced,
        [jnp.array([0.2, 0.8])],
        jnp.array([0.0, 0.0]),
        steps=0,
        rtol=1e-6,
        atol=1e-6,
        linear_solver=multistage_module.lx.QR(),
        adaptive_sampler=None,
        checkpoint_path="fake",
        checkpoint_every=0,
    )

    assert eqx.tree_equal(trained, net)


def test_trust_region_zero_steps_returns_without_solver(monkeypatch):
    """A zero-step trust-region phase should be a no-op."""
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

    def fail_least_squares(*args, **kwargs):
        del args, kwargs
        raise AssertionError("least_squares should not be called")

    monkeypatch.setattr(
        multistage_module.optimistix, "least_squares", fail_least_squares
    )

    def loss_fun_unreduced(net, x_data, u_data, x_col):
        del x_col
        return jax.vmap(net)(x_data) - u_data

    trained = _trust_region_train(
        net,
        loss_fun_unreduced,
        [jnp.array([0.2, 0.8])],
        jnp.array([0.0, 0.0]),
        steps=0,
        rtol=1e-6,
        atol=1e-6,
        linear_solver=multistage_module.lx.QR(),
        adaptive_sampler=None,
        checkpoint_path=None,
    )

    assert eqx.tree_equal(trained, net)


def test_multistage_adaptive_defaults_follow_stage_data(monkeypatch):
    """Default adaptive sample counts should be based on the current stage."""
    calls = []

    def fake_adaptive_sample(
        net, residual_fun, in_size, n_candidates, n_selected, key=None, **kwargs
    ):
        del net, residual_fun, kwargs
        calls.append((n_candidates, n_selected))
        return [jnp.zeros(n_selected) for _ in range(in_size)], key

    def fake_train(
        net,
        loss_fun,
        x,
        training_samples,
        optimizer,
        steps,
        *,
        adaptive_sampler=None,
        key=None,
        return_loss_history=True,
        **kwargs,
    ):
        del loss_fun, x, training_samples, optimizer, steps, kwargs
        if adaptive_sampler is not None:
            adaptive_sampler(net, key=key)
        return (net, []) if return_loss_history else net

    def fake_stats(*args, **kwargs):
        del args, kwargs
        return jnp.array(1.0), jnp.array(0.1), jnp.array([2.0])

    monkeypatch.setattr(multistage_module, "adaptive_sample", fake_adaptive_sample)
    monkeypatch.setattr(multistage_module, "_train", fake_train)
    monkeypatch.setattr(multistage_module, "stats", fake_stats)
    monkeypatch.setattr(multistage_module, "save", lambda *args, **kwargs: None)

    net = Stage1(
        jnp.array([0.0]),
        jnp.array([1.0]),
        in_size=1,
        out_size=1,
        width_size=2,
        depth=1,
    )
    x_stage1 = [jnp.linspace(0.0, 1.0, 4)]
    y_stage1 = jnp.zeros(4)
    x_stage2 = [jnp.linspace(0.0, 1.0, 8)]
    y_stage2 = jnp.zeros(8)

    def residual_fun(model, x):
        del model
        return x, x

    def loss_fun(model, x, y, x_col):
        del model, x, y, x_col
        return jnp.array(0.0)

    multistage_module.multistage_train(
        net,
        residual_fun,
        residual_fun,
        loss_fun,
        loss_fun,
        x_stage1,
        y_stage1,
        optimizer=optax.sgd,
        steps=1,
        learning_rate=0.0,
        adaptive_sample_freq=1,
        n_stages=2,
        x_stage2=x_stage2,
        training_samples_stage2=y_stage2,
        checkpoint_every=0,
    )

    assert calls == [(40, 2), (80, 4)]


def test_multistage_train_uses_zero_inverse_param_corrections(monkeypatch):
    """Automatic stage creation should not perturb trained PDE parameters."""
    trained_nets = []

    def fake_train(
        net,
        loss_fun,
        x,
        training_samples,
        optimizer,
        steps,
        *,
        return_loss_history=True,
        **kwargs,
    ):
        del loss_fun, x, training_samples, optimizer, steps, kwargs
        trained_nets.append(net)
        return (net, []) if return_loss_history else net

    def fake_stats(*args, **kwargs):
        del args, kwargs
        return jnp.array(1.0), jnp.array(0.25), jnp.array([2.0])

    monkeypatch.setattr(multistage_module, "_train", fake_train)
    monkeypatch.setattr(multistage_module, "stats", fake_stats)
    monkeypatch.setattr(multistage_module, "save", lambda *args, **kwargs: None)

    net = Stage1(
        jnp.array([0.0]),
        jnp.array([1.0]),
        in_size=1,
        out_size=1,
        width_size=2,
        depth=1,
        params={
            "lambda_1": jnp.array([1.5]),
            "log_lambda_2": jnp.log(jnp.array([0.25])),
        },
        params_are_trainable=True,
    )

    def residual_fun(model, x):
        del model
        return x, x

    def loss_fun(model, x, y, x_col):
        del model, x, y, x_col
        return jnp.array(0.0)

    multistage_module.multistage_train(
        net,
        residual_fun,
        residual_fun,
        loss_fun,
        loss_fun,
        [jnp.linspace(0.0, 1.0, 4)],
        jnp.zeros(4),
        optimizer=optax.sgd,
        steps=1,
        learning_rate=0.0,
        adaptive_sample_freq=0,
        n_stages=2,
        checkpoint_every=0,
    )

    assert len(trained_nets) == 2
    stage2 = trained_nets[1]
    assert stage2.params.keys == ("lambda_1", "lambda_2")
    np.testing.assert_allclose(stage2.get_param("lambda_1"), jnp.array([0.0]))
    np.testing.assert_allclose(stage2.get_param("lambda_2"), jnp.array([0.0]))


def test_multistage_train_uses_custom_stage_correction_param_map(monkeypatch):
    """Users can prescribe neutral correction params for other transforms."""
    trained_nets = []

    def fake_train(
        net,
        loss_fun,
        x,
        training_samples,
        optimizer,
        steps,
        *,
        return_loss_history=True,
        **kwargs,
    ):
        del loss_fun, x, training_samples, optimizer, steps, kwargs
        trained_nets.append(net)
        return (net, []) if return_loss_history else net

    def fake_stats(*args, **kwargs):
        del args, kwargs
        return jnp.array(1.0), jnp.array(0.25), jnp.array([2.0])

    monkeypatch.setattr(multistage_module, "_train", fake_train)
    monkeypatch.setattr(multistage_module, "stats", fake_stats)
    monkeypatch.setattr(multistage_module, "save", lambda *args, **kwargs: None)

    net = Stage1(
        jnp.array([0.0]),
        jnp.array([1.0]),
        in_size=1,
        out_size=1,
        width_size=2,
        depth=1,
        params={"log_alpha": jnp.array([0.3])},
        params_are_trainable=True,
    )

    def residual_fun(model, x):
        del model
        return x, x

    def loss_fun(model, x, y, x_col):
        del model, x, y, x_col
        return jnp.array(0.0)

    multistage_module.multistage_train(
        net,
        residual_fun,
        residual_fun,
        loss_fun,
        loss_fun,
        [jnp.linspace(0.0, 1.0, 4)],
        jnp.zeros(4),
        optimizer=optax.sgd,
        steps=1,
        learning_rate=0.0,
        adaptive_sample_freq=0,
        n_stages=2,
        stage_correction_param_map={"log_alpha": "alpha"},
        checkpoint_every=0,
    )

    assert len(trained_nets) == 2
    stage2 = trained_nets[1]
    assert stage2.params.keys == ("alpha",)
    np.testing.assert_allclose(stage2.get_param("alpha"), jnp.array([0.0]))


def test_plot_loss_skips_checkpoints_without_loss_history(tmp_path, monkeypatch):
    """Trust-region checkpoints without loss history should not break plotting."""

    class FakeManager:
        def latest_step(self):
            return 0

        def restore(self, step, args):
            del step, args
            return {"trainable": jnp.array([1.0]), "loss_ref": jnp.array(1.0)}

        def close(self):
            pass

    monkeypatch.setattr(
        plot_module,
        "checkpoint_manager",
        lambda path, create=False: FakeManager(),
    )

    figname = tmp_path / "loss.pdf"
    plot_loss("fake", figname)
    assert figname.exists()


def test_plot_loss_ignores_nonpositive_loss_values(tmp_path, monkeypatch):
    """Log-scale plots should never draw zero or negative loss values."""

    class FakeManager:
        def latest_step(self):
            return 0

        def restore(self, step, args):
            del step, args
            return {"loss_history": jnp.array([0.0, 2.0, -1.0, jnp.inf])}

        def close(self):
            pass

    monkeypatch.setattr(
        plot_module,
        "checkpoint_manager",
        lambda path, create=False: FakeManager(),
    )

    figname = tmp_path / "loss.pdf"
    _, ax = plot_loss("fake", figname)

    assert figname.exists()
    for line in ax.lines:
        y_data = np.asarray(line.get_ydata())
        if y_data.size:
            assert np.all(np.isfinite(y_data))
            assert np.all(y_data > 0)


def test_plot_2d_residual_handles_zero_residual(tmp_path):
    """A zero residual should still produce a non-degenerate color scale."""

    class ZeroResidualNet:
        lb = jnp.array([0.0, -1.0])
        ub = jnp.array([1.0, 1.0])

    def zero_residual(net, t, x):
        del net, x
        return jnp.zeros_like(t), jnp.zeros_like(t)

    figname = tmp_path / "zero_residual.pdf"
    _, ax = plot_module.plot_2d_residual(
        ZeroResidualNet(), zero_residual, resolution=4, figname=figname
    )

    assert figname.exists()
    vmin, vmax = ax.collections[0].get_clim()
    assert np.isfinite(vmin)
    assert np.isfinite(vmax)
    assert vmin < 0 < vmax


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
    kwargs = dict(
        epsilon=3.0,
        kappa=jnp.array([3, 4.6]),
        width_size=8,
        depth=3,
        params_are_trainable=params_are_trainable,
    )
    net = Stage2(s1, params=params, key=key, activation=jnp.cos, **kwargs)
    filename = tmp_path / "test_save_s2.eqx"
    save(filename, net, **kwargs)
    loaded_net = load(filename, Stage2, s1=s1, activation=jnp.cos)

    assert eqx.tree_equal(loaded_net, net)
    assert loaded_net == net
    if params_are_trainable:
        trainable_params = _trainable_params_or_none(loaded_net._params)
        assert trainable_params is not None
        assert trainable_params.keys == tuple(params)
    else:
        assert _trainable_params_or_none(loaded_net._params) is None
