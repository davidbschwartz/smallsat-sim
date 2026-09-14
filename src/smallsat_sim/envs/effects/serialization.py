"""Host codecs for realized effects; decoding never samples new effects."""
from typing import Optional, cast
import jax
import jax.numpy as jnp
import numpy as np
from .actuator_kernels import BatchedFaultState
from .disturbances import DisturbanceState
from .custom import CustomEffectState, ApplyEffect, import_function

def perturbation_state_to_serializable(
    state: Optional["BatchedFaultState"],
) -> Optional[dict]:
    if isinstance(state, CustomEffectState):
        return custom_state_to_serializable(state)
    if state is None:
        return None

    to_np = lambda x: None if x is None else np.asarray(x)

    return {
        "rng": np.asarray(state.rng),
        "thruster_mask": np.asarray(state.thruster_mask),
        "failure_value": (
            None if state.failure_value is None else int(state.failure_value)
        ),
        "start_times": to_np(state.start_times),
        "max_thruster_force": to_np(state.max_thruster_force),
        "gp_x_samples": to_np(state.gp_x_samples),
        "gp_y_samples": to_np(state.gp_y_samples),
    }


def perturbation_state_from_serializable(
    payload: Optional[dict],
) -> Optional["BatchedFaultState"]:
    if payload is not None and "custom_effect" in payload:
        return custom_state_from_serializable(payload)
    if payload is None:
        return None

    to_jnp = lambda x: None if x is None else jnp.asarray(x)

    mask = jnp.asarray(payload["thruster_mask"])
    if mask.ndim != 2:
        raise ValueError("Checkpoint actuator mask must have shape (num_envs, num_actuators)")

    def actuator_array(value):
        if value is None:
            return None
        array = jnp.asarray(value)
        # Legacy vectors preferred the actuator axis when both dimensions matched.
        if array.ndim == 1 and array.shape[0] != mask.shape[1]:
            array = array[:, None]
        return jnp.broadcast_to(array, mask.shape)

    return BatchedFaultState(
        rng=jnp.asarray(payload["rng"]),
        thruster_mask=mask,
        failure_value=payload["failure_value"],
        start_times=actuator_array(payload["start_times"]),
        max_thruster_force=actuator_array(payload["max_thruster_force"]),
        gp_x_samples=to_jnp(payload["gp_x_samples"]),
        gp_y_samples=to_jnp(payload["gp_y_samples"]),
    )


def disturbance_state_to_serializable(
    state: Optional["DisturbanceState"],
) -> Optional[dict]:
    if isinstance(state, CustomEffectState):
        return custom_state_to_serializable(state)
    if state is None:
        return None
    to_np = lambda x: None if x is None else np.asarray(x)
    params_np = jax.tree_util.tree_map(to_np, state.params)
    return {
        "rng": np.asarray(state.rng),
        "active_mask": np.asarray(state.active_mask),
        "start_times": np.asarray(state.start_times),
        "params": params_np,
    }


def disturbance_state_from_serializable(
    payload: Optional[dict],
) -> Optional["DisturbanceState"]:
    if payload is not None and "custom_effect" in payload:
        return custom_state_from_serializable(payload)
    if payload is None:
        return None
    to_jnp = lambda x: None if x is None else jnp.asarray(x)
    params = jax.tree_util.tree_map(to_jnp, payload["params"])
    return DisturbanceState(
        rng=jnp.asarray(payload["rng"]),
        active_mask=jnp.asarray(payload["active_mask"]),
        start_times=jnp.asarray(payload["start_times"]),
        params=params,
    )


def custom_state_to_serializable(state: CustomEffectState) -> dict:
    """Save the realization and implementation identity for replay."""
    return {
        "custom_effect": state.apply_path,
        "dt": state.dt,
        "version": state.version,
        "data": jax.tree.map(np.asarray, state.data),
        "active_mask": np.asarray(state.active_mask),
        "start_times": np.asarray(state.start_times),
    }


def custom_state_from_serializable(payload: dict) -> CustomEffectState:
    """Restore arrays and resolve the apply function without sampling again."""
    path = payload["custom_effect"]
    return CustomEffectState(
        data=jax.tree.map(jnp.asarray, payload["data"]),
        active_mask=jnp.asarray(payload["active_mask"]),
        start_times=jnp.asarray(payload["start_times"]),
        apply_path=path,
        dt=payload["dt"],
        apply_fn=cast(ApplyEffect, import_function(path)),
        version=payload["version"],
    )

