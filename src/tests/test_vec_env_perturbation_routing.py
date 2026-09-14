"""Vectorized environment perturbation routing tests."""

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import pytest

from smallsat_sim.envs.vec_env import mjx_backend as vec_env
from smallsat_sim.envs.effects.actuator_kernels import (
    BatchedFaultState,
    PerturbationStatus,
)


@dataclass
class DummyBatch:
    qfrc_applied: jnp.ndarray
    time: jnp.ndarray

    @property
    def xmat(self):
        return jnp.broadcast_to(jnp.eye(3), (self.time.shape[0], 2, 3, 3))


def _build_state(
    *,
    failure_value: int,
    thruster_mask_value: int,
    num_envs: int,
    num_thrusters: int,
) -> BatchedFaultState:
    thruster_mask = jnp.full(
        (num_envs, num_thrusters), thruster_mask_value, dtype=jnp.int32
    )
    start_times = jnp.zeros((num_envs, num_thrusters), dtype=jnp.float32)
    gp_x_samples = jnp.tile(
        jnp.array([0.0, 0.5, 1.0], dtype=jnp.float32), (num_thrusters, 1)
    )
    gp_y_samples = jnp.tile(
        jnp.array([0.0, 0.2, 0.8], dtype=jnp.float32), (num_thrusters, 1)
    )
    max_thruster_force = jnp.full(
        (num_envs, num_thrusters), 1.0, dtype=jnp.float32
    )
    return BatchedFaultState(
        rng=jax.random.PRNGKey(0),
        thruster_mask=thruster_mask,
        failure_value=failure_value,
        start_times=start_times,
        max_thruster_force=max_thruster_force,
        gp_x_samples=gp_x_samples,
        gp_y_samples=gp_y_samples,
    )


@pytest.mark.parametrize(
    "failure_value,expected_force",
    [
        (PerturbationStatus.STUCK_OFF.value, 0.0),
        (PerturbationStatus.STUCK_ON.value, 1.0),
        (PerturbationStatus.FAULTY_VALVE.value, 0.04),
        (PerturbationStatus.SATURATED_THRUST.value, 0.04),
        (PerturbationStatus.THRUST_INSTABILITY.value, 0.04),
    ],
)
def test_prepare_effects_routes_failure_modes(
    failure_value: int, expected_force: float
) -> None:
    num_envs = 1
    num_thrusters = 2
    base_ctrl = jnp.array([[0.1, 0.9]], dtype=jnp.float32)
    dummy_batch = DummyBatch(
        qfrc_applied=jnp.zeros((num_envs, 6)),
        time=jnp.array([1.0], dtype=jnp.float32),
    )
    pert_state = _build_state(
        failure_value=failure_value,
        thruster_mask_value=failure_value,
        num_envs=num_envs,
        num_thrusters=num_thrusters,
    )
    pert_state = replace(pert_state, thruster_mask=pert_state.thruster_mask.at[:, 1].set(0))
    env_state = vec_env.VecEnvState(
        rng=jax.random.PRNGKey(1),
        mjx_batch=dummy_batch,
        terminal_hold_counts=jnp.zeros((num_envs,), dtype=jnp.int32),
        disturbance_states=(),
        perturbation_states=(pert_state,),
    )

    _, applied_ctrl, _ = vec_env.prepare_effects(env_state, base_ctrl)

    # Only the first actuator is faulty. The GP table maps 0.1 to 0.04.
    assert jnp.allclose(applied_ctrl, jnp.array([[expected_force, 0.9]]))


def test_prepare_effects_fallback_applies_masked_stuck_off() -> None:
    num_envs = 1
    num_thrusters = 2
    base_ctrl = jnp.array([[0.3, 0.7]], dtype=jnp.float32)
    dummy_batch = DummyBatch(
        qfrc_applied=jnp.zeros((num_envs, 6)),
        time=jnp.array([1.0], dtype=jnp.float32),
    )
    pert_state = _build_state(
        failure_value=999,
        thruster_mask_value=PerturbationStatus.STUCK_OFF.value,
        num_envs=num_envs,
        num_thrusters=num_thrusters,
    )
    env_state = vec_env.VecEnvState(
        rng=jax.random.PRNGKey(2),
        mjx_batch=dummy_batch,
        terminal_hold_counts=jnp.zeros((num_envs,), dtype=jnp.int32),
        disturbance_states=(),
        perturbation_states=(pert_state,),
    )

    _, applied_ctrl, _ = vec_env.prepare_effects(env_state, base_ctrl)

    assert jnp.allclose(applied_ctrl, jnp.zeros_like(base_ctrl))
