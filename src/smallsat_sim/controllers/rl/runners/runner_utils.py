"""Checkpoint, metadata and array helpers for RL runners."""

from collections.abc import Mapping, Sequence
from pathlib import Path
import hashlib

from flax import nnx
from mujoco import mjx
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from smallsat_sim.envs.effects.disturbances import (
    DisturbanceState,
)
from smallsat_sim.envs.effects.actuator_kernels import (
    BatchedFaultState,
)
from smallsat_sim.envs.vec_env.types import VecEnvState
from smallsat_sim.envs.vec_env.serialization import vecenv_state_from_serializable, vecenv_state_to_serializable
from smallsat_sim.envs.effects.serialization import (
    disturbance_state_from_serializable, disturbance_state_to_serializable,
    perturbation_state_from_serializable, perturbation_state_to_serializable,
)

_serialization_type_key = "__smallsat_type__"
_schema_version_key = "__smallsat_schema_version__"
_schema_version = 2


def _is_jax_array(x):
    return isinstance(x, (jnp.ndarray, jax.Array))


def _empty_array_payload(obj):
    array = np.asarray(obj)
    if array.size != 0:
        return None
    return {
        _serialization_type_key: "EmptyArray",
        "shape": np.asarray(array.shape, dtype=np.int32),
        "dtype": str(array.dtype),
    }


def _typed_payload(kind: str, payload):
    return {
        _serialization_type_key: kind,
        "payload": payload,
    }


def _special_checkpoint_payload(obj):
    empty_payload = None
    if _is_jax_array(obj) or isinstance(obj, np.ndarray):
        empty_payload = _empty_array_payload(obj)
    if empty_payload is not None:
        return empty_payload
    if isinstance(obj, VecEnvState):
        return _typed_payload("VecEnvState", vecenv_state_to_serializable(obj))
    if isinstance(obj, DisturbanceState):
        return _typed_payload(
            "DisturbanceState", disturbance_state_to_serializable(obj)
        )
    if isinstance(obj, BatchedFaultState):
        return _typed_payload(
            "BatchedFaultState", perturbation_state_to_serializable(obj)
        )
    return None


def _prepare_for_checkpoint(obj):
    if isinstance(obj, nnx.State):
        obj = nnx.to_pure_dict(obj)
    special_payload = _special_checkpoint_payload(obj)
    if special_payload is not None:
        return _prepare_for_checkpoint(special_payload)
    if _is_jax_array(obj):
        return np.asarray(obj)
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, Mapping):
        return {k: _prepare_for_checkpoint(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return _typed_payload("Tuple", [_prepare_for_checkpoint(v) for v in obj])
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return [_prepare_for_checkpoint(v) for v in obj]
    return obj


def _restore_typed_payload(obj, *, mjx_batch_template: mjx.Data | None = None):
    """Reconstruct a tagged checkpoint value, using an MJX template for environment state."""
    kind = obj[_serialization_type_key]
    if kind == "EmptyArray":
        return np.empty(
            tuple(int(x) for x in obj["shape"]),
            dtype=np.dtype(obj["dtype"]),
        )

    payload = _restore_from_serializable(
        obj["payload"], mjx_batch_template=mjx_batch_template
    )
    if kind == "Tuple":
        return tuple(payload)
    if kind == "VecEnvState":
        if mjx_batch_template is None:
            return payload
        return vecenv_state_from_serializable(
            payload, mjx_batch_template=mjx_batch_template
        )
    if kind == "DisturbanceState":
        return disturbance_state_from_serializable(payload)
    if kind == "BatchedFaultState":
        return perturbation_state_from_serializable(payload)
    return obj


def _restore_from_serializable(obj, *, mjx_batch_template: mjx.Data | None = None):
    """Recursively restore containers and tagged values from checkpoint data."""
    if isinstance(obj, Mapping):
        if _serialization_type_key in obj:
            return _restore_typed_payload(obj, mjx_batch_template=mjx_batch_template)
        return {
            k: _restore_from_serializable(v, mjx_batch_template=mjx_batch_template)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [
            _restore_from_serializable(v, mjx_batch_template=mjx_batch_template)
            for v in obj
        ]
    if isinstance(obj, tuple):
        return tuple(
            _restore_from_serializable(v, mjx_batch_template=mjx_batch_template)
            for v in obj
        )
    return obj


def _to_jnp_recursive(obj):
    if isinstance(obj, np.ndarray):
        return jnp.asarray(obj)
    if isinstance(obj, (VecEnvState, DisturbanceState, BatchedFaultState)):
        return obj
    if isinstance(obj, Mapping):
        return {k: _to_jnp_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jnp_recursive(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_jnp_recursive(v) for v in obj)
    return obj


def resolve_checkpoint_paths(checkpoint, *, stage, policy_filename, adaptation_filename):
    """Use the exact supplied checkpoint and the default sibling for its companion."""
    checkpoint = Path(checkpoint)
    policy = checkpoint.parent / policy_filename
    adaptation = checkpoint.parent / adaptation_filename
    if stage == "policy":
        policy = checkpoint
    elif stage == "adaptation":
        adaptation = checkpoint
    else:
        raise ValueError("Expected a policy or adaptation checkpoint")
    return policy, adaptation


def checkpoint_exists(directory, filename):
    return (Path(directory) / filename / "_METADATA").is_file()


def _save(directory, filename, payload):
    path = (Path(directory) / filename).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {_schema_version_key: _schema_version, **payload}
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(path, _prepare_for_checkpoint(payload), force=True)


def load_trained_modules(directory, filename, *, mjx_batch_template=None):
    path = (Path(directory) / filename).resolve()
    if not checkpoint_exists(directory, filename):
        raise FileNotFoundError(f"New-format checkpoint not found: {path}")
    with ocp.PyTreeCheckpointer() as checkpointer:
        payload = checkpointer.restore(path)
    if payload.pop(_schema_version_key, None) != _schema_version:
        raise ValueError("Unsupported checkpoint schema; train a new model")
    return _to_jnp_recursive(
        _restore_from_serializable(payload, mjx_batch_template=mjx_batch_template)
    )


def save_training_data(directory, filename, data):
    _save(directory, filename, {"data": data})


def load_training_data(directory, filename, *, mjx_batch_template=None):
    return load_trained_modules(
        directory,
        filename,
        mjx_batch_template=mjx_batch_template,
    )["data"]


def _extract_optimizer_state(optimizer):
    return (
        nnx.state(optimizer, nnx.optimizer.OptState) if optimizer is not None else None
    )


def _checked_state(template, source):
    """Validate all keys, shapes and dtypes before mutating any live model."""
    target = nnx.to_pure_dict(template)
    if isinstance(source, nnx.State):
        source = nnx.to_pure_dict(source)

    def check(expected, actual, path="state"):
        if isinstance(expected, Mapping):
            if not isinstance(
                actual,
                Mapping,
            ) or {
                str(k) for k in expected
            } != {str(k) for k in actual}:
                raise ValueError(f"Checkpoint keys differ at {path}")
            normalized = {str(k): v for k, v in actual.items()}
            return {
                k: check(
                    v,
                    normalized[str(k)],
                    f"{path}/{k}",
                )
                for k, v in expected.items()
            }
        expected, actual = jnp.asarray(expected), jnp.asarray(actual)
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            raise ValueError(f"Checkpoint shape/dtype differ at {path}")
        return jax.device_put(actual, expected.sharding)

    restored = check(target, source)
    nnx.replace_by_pure_dict(template, restored)
    return template


def update_module_from_checkpoint_state(module, checkpoint_state):
    nnx.update(module, _checked_state(nnx.state(module), checkpoint_state))


def _restore_optimizer(optimizer, state):
    nnx.update(optimizer, _checked_state(_extract_optimizer_state(optimizer), state))


def save_trained_modules(agent, directory, filename, *, metadata=None):
    _save(
        directory,
        filename,
        {
            "actor_model": nnx.state(agent.actor),
            "critic_model": nnx.state(agent.critic),
            "actor_optimizer": _extract_optimizer_state(agent.actor_optimizer),
            "critic_optimizer": _extract_optimizer_state(agent.critic_optimizer),
            "agent_rng": agent.key,
            "metadata": metadata or {},
        },
    )


def restore_trained_modules(agent, directory, filename):
    payload = load_trained_modules(directory, filename)
    # Validate everything before applying any state.
    objects = (agent.actor, agent.critic, agent.actor_optimizer, agent.critic_optimizer)
    templates = (
        nnx.state(agent.actor),
        nnx.state(agent.critic),
        _extract_optimizer_state(agent.actor_optimizer),
        _extract_optimizer_state(agent.critic_optimizer),
    )
    names = ("actor_model", "critic_model", "actor_optimizer", "critic_optimizer")
    states = [
        _checked_state(
            template,
            payload[name],
        )
        for template, name in zip(templates, names)
    ]
    for obj, state in zip(objects, states):
        nnx.update(obj, state)
    agent.key = jax.device_put(payload["agent_rng"], agent.key.sharding)
    return payload


def save_adaptation_module(
    am,
    directory,
    filename,
    *,
    optimizer=None,
    rng_key=None,
    metadata=None,
):
    _save(
        directory,
        filename,
        {
            "am_model": nnx.state(am),
            "am_optimizer": _extract_optimizer_state(optimizer),
            "am_rng": rng_key,
            "metadata": metadata or {},
        },
    )


def model_fingerprint(model):
    """Bind an estimator checkpoint to the exact teacher weights it was trained with."""
    digest = hashlib.sha256()
    for value in jax.tree.leaves(nnx.state(model)):
        digest.update(np.asarray(value).tobytes())
    return digest.hexdigest()
