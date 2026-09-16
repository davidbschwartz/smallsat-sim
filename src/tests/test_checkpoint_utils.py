"""Checkpoint discovery and training-data persistence tests."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import jax.numpy as jnp

from smallsat_sim.controllers.rl.runners.runner_utils import (
    checkpoint_exists,
    load_training_data,
    save_training_data,
)


def test_training_data_checkpoint_roundtrip(tmp_path: Path) -> None:
    checkpoint_name = "roundtrip_test.ckpt"
    payload = {
        "obs": jnp.arange(6, dtype=jnp.float32).reshape(2, 3),
        "nested": [
            np.array([1, 2], dtype=np.int32),
            {"value": jnp.asarray([3.0, 4.0], dtype=jnp.float32)},
        ],
    }

    save_training_data(str(tmp_path) + "/", checkpoint_name, payload)

    assert checkpoint_exists(str(tmp_path) + "/", checkpoint_name)

    restored = load_training_data(str(tmp_path) + "/", checkpoint_name)
    np.testing.assert_array_equal(
        np.asarray(restored["obs"]), np.asarray(payload["obs"])
    )
    np.testing.assert_array_equal(
        np.asarray(restored["nested"][0]), np.asarray(payload["nested"][0])
    )
    np.testing.assert_array_equal(
        np.asarray(restored["nested"][1]["value"]),
        np.asarray(payload["nested"][1]["value"]),
    )


def test_training_data_checkpoint_preserves_empty_arrays_and_sequences(
    tmp_path: Path,
) -> None:
    checkpoint_name = "empty_arrays.ckpt"
    payload = {
        "empty_jax": jnp.empty((0, 3), dtype=jnp.float32),
        "empty_numpy": np.empty((2, 0), dtype=np.int32),
        "tuple_value": (jnp.asarray([1.0]), np.asarray([2], dtype=np.int32)),
        "list_value": [np.asarray([3.0]), jnp.asarray([4.0])],
    }

    save_training_data(str(tmp_path) + "/", checkpoint_name, payload)
    restored = load_training_data(str(tmp_path) + "/", checkpoint_name)

    assert restored["empty_jax"].shape == (0, 3)
    assert restored["empty_jax"].dtype == jnp.float32
    assert restored["empty_numpy"].shape == (2, 0)
    assert restored["empty_numpy"].dtype == jnp.int32
    assert isinstance(restored["tuple_value"], tuple)
    assert isinstance(restored["list_value"], list)
    np.testing.assert_array_equal(np.asarray(restored["tuple_value"][0]), [1.0])
    np.testing.assert_array_equal(np.asarray(restored["tuple_value"][1]), [2])
    np.testing.assert_array_equal(np.asarray(restored["list_value"][0]), [3.0])
    np.testing.assert_array_equal(np.asarray(restored["list_value"][1]), [4.0])


def test_training_data_checkpoint_overwrites_existing_directory(tmp_path: Path) -> None:
    checkpoint_name = "overwrite.ckpt"

    save_training_data(
        str(tmp_path) + "/", checkpoint_name, {"value": jnp.asarray([1])}
    )
    save_training_data(
        str(tmp_path) + "/", checkpoint_name, {"value": jnp.asarray([2])}
    )

    restored = load_training_data(str(tmp_path) + "/", checkpoint_name)
    np.testing.assert_array_equal(np.asarray(restored["value"]), [2])


def _checkpoint_agent(seed):
    import jax
    import optax
    from flax import nnx

    actor = nnx.Sequential(nnx.Linear(2, 2, rngs=nnx.Rngs(seed)))
    critic = nnx.Linear(2, 1, rngs=nnx.Rngs(seed + 1))
    return SimpleNamespace(
        actor=actor,
        critic=critic,
        actor_optimizer=nnx.Optimizer(actor, optax.adam(0.01)),
        critic_optimizer=nnx.Optimizer(critic, optax.adam(0.01)),
        key=jax.random.PRNGKey(seed),
    )


def _update_checkpoint_agent(agent):
    from flax import nnx

    for model, optimizer in (
        (agent.actor, agent.actor_optimizer),
        (agent.critic, agent.critic_optimizer),
    ):
        grads = nnx.grad(lambda m: jnp.square(m(jnp.ones((3, 2)))).mean())(model)
        optimizer.update(grads)


def test_model_optimizer_rng_roundtrip_and_next_update(tmp_path):
    import jax
    from flax import nnx
    from smallsat_sim.controllers.rl.runners.runner_utils import (
        restore_trained_modules,
        save_trained_modules,
    )

    original = _checkpoint_agent(1)
    _update_checkpoint_agent(original)
    save_trained_modules(original, tmp_path, "agent.ckpt")
    assert (tmp_path / "agent.ckpt" / "_METADATA").is_file()
    assert not (tmp_path / "agent.ckpt" / "payload.pkl").exists()
    restored = _checkpoint_agent(10)
    restore_trained_modules(restored, tmp_path, "agent.ckpt")
    np.testing.assert_array_equal(original.key, restored.key)
    for attr in ("actor", "critic", "actor_optimizer", "critic_optimizer"):
        jax.tree.map(
            np.testing.assert_array_equal,
            nnx.state(getattr(original, attr)),
            nnx.state(getattr(restored, attr)),
        )
    _update_checkpoint_agent(original)
    _update_checkpoint_agent(restored)
    for attr in ("actor", "critic"):
        jax.tree.map(
            np.testing.assert_array_equal,
            nnx.state(getattr(original, attr)),
            nnx.state(getattr(restored, attr)),
        )


def test_incompatible_state_rejected_without_mutation():
    import pytest
    from flax import nnx
    from smallsat_sim.controllers.rl.runners.runner_utils import update_module_from_checkpoint_state
    model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    before = np.asarray(model.kernel).copy()
    bad = {"kernel": jnp.zeros((3, 3)), "bias": jnp.zeros(3)}
    with pytest.raises(ValueError, match="shape/dtype"):
        update_module_from_checkpoint_state(model, bad)
    np.testing.assert_array_equal(model.kernel, before)
    with pytest.raises(ValueError, match="keys"):
        update_module_from_checkpoint_state(model, {"kernel": model.kernel.value})


def test_adaptation_module_roundtrip(tmp_path):
    from smallsat_sim.controllers.rl.runners.runner_utils import (
        load_trained_modules,
        save_adaptation_module,
        update_module_from_checkpoint_state,
    )

    original = _checkpoint_agent(1)
    save_adaptation_module(original.actor, tmp_path, "am.ckpt")
    restored = _checkpoint_agent(10)
    state = load_trained_modules(tmp_path, "am.ckpt")
    update_module_from_checkpoint_state(restored.actor, state["am_model"])
    np.testing.assert_array_equal(original.actor(jnp.ones((1, 2))),
                                  restored.actor(jnp.ones((1, 2))))
    assert state["am_optimizer"] is None


def test_save_error_propagates_without_pickle_fallback(tmp_path, monkeypatch):
    import orbax.checkpoint as ocp
    import pytest

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(ocp.PyTreeCheckpointer, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        save_training_data(tmp_path, "failed.ckpt", {"value": jnp.ones(1)})
    assert not checkpoint_exists(tmp_path, "failed.ckpt")
