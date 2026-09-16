"""Pure actuator-effect kernels for vectorized simulations."""

from smallsat_sim.envs.effects.custom import (
    CustomEffectState,
)
from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional, Tuple

import jax
import jax.numpy as jnp


class PerturbationStatus(Enum):
    OPERATIONAL = 0
    STUCK_OFF = 1
    STUCK_ON = 2
    FAULTY_VALVE = 3
    SATURATED_THRUST = 4
    THRUST_INSTABILITY = 5


@dataclass
class BatchedFaultState:
    """Sampled actuator data consumed by pure rollout kernels and checkpoints."""

    rng: jnp.ndarray
    thruster_mask: jnp.ndarray
    failure_value: Optional[int] = None
    start_times: Optional[jnp.ndarray] = None
    max_thruster_force: Optional[jnp.ndarray] = None
    gp_x_samples: Optional[jnp.ndarray] = None
    gp_y_samples: Optional[jnp.ndarray] = None


# failure_value is static dispatch metadata, not a traced array.
jax.tree_util.register_dataclass(
    BatchedFaultState,
    data_fields=["rng", "thruster_mask", "start_times", "max_thruster_force",
                 "gp_x_samples", "gp_y_samples"],
    meta_fields=["failure_value"],
)


def _active(state, control, timestamp, status):
    """Controls and actuator state are (environments, actuators); time is per row."""
    if control.ndim != 2 or state.thruster_mask.shape != control.shape:
        raise ValueError("control and actuator mask must have shape (num_envs, num_actuators)")
    if state.start_times is not None and state.start_times.shape != control.shape:
        raise ValueError("Actuator onset times must match control shape")
    time = jnp.asarray(timestamp)
    if time.shape not in ((), (control.shape[0],), (control.shape[0], 1)):
        raise ValueError("timestamp must be scalar or have one value per environment")
    if time.ndim == 1:
        time = time[:, None]
    starts = 0.0 if state.start_times is None else state.start_times
    return (state.thruster_mask == status) & (time >= starts)


def stuck_off_apply_from_state(
    state: Optional[BatchedFaultState],
    control: jnp.ndarray,
    timestamp: float,
) -> Tuple[jnp.ndarray, Optional[BatchedFaultState]]:
    """Zero scheduled actuators whose onset has passed; leave fault state unchanged."""
    if state is None:
        return control, state
    control = jnp.asarray(control)
    active = _active(state, control, timestamp, PerturbationStatus.STUCK_OFF.value)
    control = jnp.where(active, 0.0, control)
    return control, state


def stuck_on_apply_from_state(
    state: Optional[BatchedFaultState],
    control: jnp.ndarray,
    timestamp: float,
) -> Tuple[jnp.ndarray, Optional[BatchedFaultState]]:
    """Replace active commands with their sampled held force."""
    if state is None:
        return control, state

    start_times = state.start_times
    max_force = state.max_thruster_force
    if start_times is None or max_force is None:
        return control, state

    control = jnp.asarray(control)
    active = _active(state, control, timestamp, PerturbationStatus.STUCK_ON.value)
    control = jnp.where(active, max_force, control)
    return control, state


def gp_apply_from_state(
    state: Optional[BatchedFaultState],
    control: jnp.ndarray,
    timestamp: float,
    failure_value: int | None = None,
) -> Tuple[jnp.ndarray, Optional[BatchedFaultState]]:
    """Transform active commands through prepared curves; no sampling occurs here."""
    if (
        state is None
        or state.start_times is None
        or state.gp_x_samples is None
        or state.gp_y_samples is None
    ):
        return control, state

    control = jnp.asarray(control)
    original_control = control
    gp_x_samples = state.gp_x_samples
    gp_y_samples = state.gp_y_samples

    active = _active(state, control, timestamp, state.failure_value if failure_value is None else failure_value)
    gp_support = jnp.any(jnp.diff(gp_x_samples, axis=-1) != 0, axis=-1)
    active = active & gp_support

    # Fast fixed-grid interpolation. GP registration/precomputation stores each
    # failed-thruster curve as a monotone lookup table, so runtime application
    # only needs two gathers and a linear blend instead of per-element jnp.interp.
    n_points = gp_y_samples.shape[-1]
    x_min_b = gp_x_samples[..., 0]
    x_max_b = gp_x_samples[..., -1]
    scale = (n_points - 1) / jnp.maximum(x_max_b - x_min_b, 1e-6)
    grid_pos = (control - x_min_b) * scale
    grid_pos = jnp.clip(grid_pos, 0.0, float(n_points - 1))
    idx0 = jnp.floor(grid_pos).astype(jnp.int32)
    idx1 = jnp.minimum(idx0 + 1, n_points - 1)
    frac = grid_pos - idx0.astype(grid_pos.dtype)

    thruster_idx = jnp.arange(control.shape[-1], dtype=jnp.int32)[None, :]
    if gp_y_samples.ndim == 3:
        # Independent episode resets may retain a different curve in each lane.
        env_idx = jnp.arange(control.shape[0], dtype=jnp.int32)[:, None]
        y0 = gp_y_samples[env_idx, thruster_idx, idx0]
        y1 = gp_y_samples[env_idx, thruster_idx, idx1]
    else:
        y0 = gp_y_samples[thruster_idx, idx0]
        y1 = gp_y_samples[thruster_idx, idx1]
    interp_control = y0 + frac * (y1 - y0)
    interp_control = jnp.nan_to_num(
        interp_control,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    control = jnp.where(active, interp_control, control)
    control = jnp.where(jnp.isnan(control), original_control, control)

    max_force = state.max_thruster_force
    if max_force is not None:
        # Retain the optional physical envelope recorded in older snapshots.
        control = jnp.clip(control, -max_force, max_force)

    return control, state


def gp_register_state(
    state: BatchedFaultState,
    thruster_index: int,
    x_samples: jnp.ndarray,
    y_samples: jnp.ndarray,
) -> BatchedFaultState:
    num_thrusters = state.thruster_mask.shape[1]
    num_points = x_samples.shape[0]
    sanitize = lambda arr: jnp.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    x_samples = sanitize(x_samples)
    y_samples = sanitize(y_samples)

    if state is not None and state.gp_x_samples is not None:
        gp_x = state.gp_x_samples
        gp_y = state.gp_y_samples
    else:
        gp_x = jnp.zeros((num_thrusters, num_points), dtype=x_samples.dtype)
        gp_y = jnp.zeros_like(gp_x)

    gp_x = gp_x.at[thruster_index].set(x_samples)
    gp_y = gp_y.at[thruster_index].set(y_samples)
    gp_x = sanitize(gp_x)
    gp_y = sanitize(gp_y)

    return replace(state, gp_x_samples=gp_x, gp_y_samples=gp_y)


def __getattr__(name):
    # Keep old checkpoint imports lazy.
    if name in ('perturbation_state_to_serializable', 'perturbation_state_from_serializable'):
        from . import serialization
        return getattr(serialization, name)
    raise AttributeError(name)
