"""Host serialization of vector state, composed with effect codecs."""
from dataclasses import fields, is_dataclass
import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx
from .types import VecEnvState
from smallsat_sim.envs.effects.serialization import (
    disturbance_state_to_serializable, disturbance_state_from_serializable,
    perturbation_state_to_serializable, perturbation_state_from_serializable,
)

def vecenv_state_to_serializable(state: "VecEnvState") -> dict:
    """
    Convert a VecEnvState into host-serializable numpy-backed payload.
    """

    def to_numpy(value):
        if is_dataclass(value):
            return {
                field.name: to_numpy(getattr(value, field.name))
                for field in fields(value)
            }
        if isinstance(value, (jax.Array, np.ndarray)):
            return np.asarray(value)
        if isinstance(value, tuple):
            return tuple(to_numpy(item) for item in value)
        return value

    mjx_payload = to_numpy(state.mjx_batch)

    return {
        "rng": np.asarray(state.rng),
        "mjx": mjx_payload,
        "terminal_hold_counts": np.asarray(state.terminal_hold_counts),
        "disturbance_states": [
            disturbance_state_to_serializable(s) for s in state.disturbance_states
        ],
        "perturbation_states": [
            perturbation_state_to_serializable(s) for s in state.perturbation_states
        ],
    }


def vecenv_state_from_serializable(
    payload: dict,
    *,
    mjx_batch_template: mjx.Data,
) -> "VecEnvState":
    """
    Reconstruct a VecEnvState from the serialized payload using the provided mjx
    template (typically the environment's freshly reset batch).
    """

    def restore(template, value):
        if is_dataclass(template):
            return template.replace(
                **{
                    field.name: restore(
                        getattr(template, field.name), value[field.name]
                    )
                    for field in fields(template)
                }
            )
        if isinstance(template, np.ndarray):
            return np.asarray(value)
        if isinstance(template, jax.Array):
            return jnp.asarray(value)
        if isinstance(template, tuple):
            return tuple(restore(t, v) for t, v in zip(template, value, strict=True))
        return value

    mjx_batch = restore(mjx_batch_template, payload["mjx"])

    disturbance_states = tuple(
        disturbance_state_from_serializable(s) if s is not None else None
        for s in payload["disturbance_states"]
    )
    perturbation_states = tuple(
        perturbation_state_from_serializable(s) if s is not None else None
        for s in payload["perturbation_states"]
    )

    return VecEnvState(
        rng=jnp.asarray(payload["rng"]),
        mjx_batch=mjx_batch,
        terminal_hold_counts=jnp.asarray(
            payload.get(
                "terminal_hold_counts",
                np.zeros((mjx_batch.qpos.shape[0],), dtype=np.int32),
            )
        ),
        disturbance_states=disturbance_states,
        perturbation_states=perturbation_states,
    )

