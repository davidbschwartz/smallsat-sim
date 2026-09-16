"""Dataclasses for vectorized simulator state and outputs."""

from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from mujoco import mjx

from smallsat_sim.envs.effects.disturbances import (
    DisturbanceState,
)
from smallsat_sim.envs.effects.actuator_kernels import (
    BatchedFaultState,
)


@dataclass
class VecEnvState:
    """Complete dynamic MJX state, including RNG and realized effects."""

    rng: jnp.ndarray
    mjx_batch: mjx.Data
    terminal_hold_counts: jnp.ndarray
    disturbance_states: Tuple[Optional[DisturbanceState], ...] = ()
    perturbation_states: Tuple[Optional[BatchedFaultState], ...] = ()

    def replace(self, **updates) -> "VecEnvState":
        return replace(self, **updates)


@dataclass
class FreeFlyerVecEnvState:
    """Compact physics and effect state for free-flyer rollouts."""

    rng: jnp.ndarray
    qpos: jnp.ndarray
    vel_body: jnp.ndarray
    omega: jnp.ndarray
    time: jnp.ndarray
    ctrl: jnp.ndarray
    actuator_force: jnp.ndarray
    terminal_hold_counts: jnp.ndarray
    disturbance_states: Tuple[Optional[DisturbanceState], ...] = ()
    perturbation_states: Tuple[Optional[BatchedFaultState], ...] = ()

    def replace(self, **updates) -> "FreeFlyerVecEnvState":
        return replace(self, **updates)


@dataclass
class VecEnvStepOutput:
    """Full transition output, registered as a JAX PyTree."""

    prev_states: jnp.ndarray
    next_states: jnp.ndarray
    rewards: jnp.ndarray
    terminals: jnp.ndarray
    commanded_ctrl: jnp.ndarray
    applied_ctrl: jnp.ndarray
    actual_wrench: jnp.ndarray
    desired_wrench: jnp.ndarray
    prev_obs: jnp.ndarray
    next_obs: jnp.ndarray
    success_terminals: jnp.ndarray
    failure_terminals: jnp.ndarray
    reward_components: Dict[str, jnp.ndarray]


@dataclass
class VecEnvTrainingStepOutput:
    """Training output without absolute observations; reward components are optional."""

    prev_states: jnp.ndarray
    next_position_error: jnp.ndarray
    next_attitude_error: jnp.ndarray
    next_speed: jnp.ndarray
    next_angular_speed: jnp.ndarray
    rewards: jnp.ndarray
    terminals: jnp.ndarray
    applied_ctrl: jnp.ndarray
    actual_wrench: jnp.ndarray
    success_terminals: jnp.ndarray
    failure_terminals: jnp.ndarray
    reward_components: Dict[str, jnp.ndarray]


# Every state/output field is dynamic PyTree data, in dataclass declaration order.
for state_type in (VecEnvState, FreeFlyerVecEnvState, VecEnvStepOutput, VecEnvTrainingStepOutput):
    jax.tree_util.register_dataclass(state_type)
