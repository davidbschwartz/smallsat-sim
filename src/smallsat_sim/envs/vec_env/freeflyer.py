"""Unconstrained free-body dynamics and explicit-state transitions."""
from .observations import state_features as observation_features
from smallsat_sim.envs.effects.wrenches import apply_world_wrenches
from .reset import sample_initial_conditions, merge_reset_rows, validate_reset_mask
from .transition import evaluate_transition, training_output, full_output
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from mujoco import mjx

from smallsat_sim.envs.effects.catalog import apply_actuator_effects
from smallsat_sim.utils.quaternions_jax import (
    quaternion_rate_matrix,
    quaternion_to_rotation_matrix,
)
from .config import VecEnvStepConfig
from smallsat_sim.envs.vec_env.types import FreeFlyerVecEnvState, VecEnvState, VecEnvStepOutput


def state_features(
    state: FreeFlyerVecEnvState,
    next_waypoint: jnp.ndarray,
) -> jnp.ndarray:
    return observation_features(state.qpos, state.vel_body, state.omega, next_waypoint)


def observations(state: FreeFlyerVecEnvState) -> jnp.ndarray:
    return jnp.concatenate((state.qpos, state.vel_body, state.omega), axis=1)


def from_mjx(state: VecEnvState) -> FreeFlyerVecEnvState:
    rot_world_to_body = jnp.swapaxes(state.mjx_batch.xmat[:, 1, :, :], 1, 2)
    vel_body = jnp.einsum("bij,bj->bi", rot_world_to_body, state.mjx_batch.qvel[:, :3])
    return FreeFlyerVecEnvState(
        rng=state.rng,
        qpos=state.mjx_batch.qpos,
        vel_body=vel_body,
        omega=state.mjx_batch.qvel[:, 3:6],
        time=state.mjx_batch.time,
        ctrl=state.mjx_batch.ctrl,
        actuator_force=state.mjx_batch.actuator_force,
        terminal_hold_counts=state.terminal_hold_counts,
        disturbance_states=state.disturbance_states,
        perturbation_states=state.perturbation_states,
    )


def to_mjx(
    state: FreeFlyerVecEnvState,
    template_state: VecEnvState,
    config: VecEnvStepConfig,
) -> VecEnvState:
    rot_body_to_world = quaternion_to_rotation_matrix(state.qpos[:, 3:7])
    vel_world = jnp.einsum("bij,bj->bi", rot_body_to_world, state.vel_body)
    qvel = jnp.concatenate((vel_world, state.omega), axis=1)
    mjx_batch = template_state.mjx_batch.replace(
        qpos=state.qpos,
        qvel=qvel,
        time=state.time,
        ctrl=state.ctrl,
        actuator_force=state.actuator_force,
    )
    mjx_batch = jax.vmap(mjx.forward, in_axes=(None, 0))(config.mjx_model, mjx_batch)
    mjx_batch = mjx_batch.replace(
        time=state.time,
        ctrl=state.ctrl,
        actuator_force=state.actuator_force,
    )
    return VecEnvState(
        rng=state.rng,
        mjx_batch=mjx_batch,
        terminal_hold_counts=state.terminal_hold_counts,
        disturbance_states=state.disturbance_states,
        perturbation_states=state.perturbation_states,
    )


def reset_from_key(
    rng_key: jnp.ndarray,
    *,
    config: VecEnvStepConfig,
) -> FreeFlyerVecEnvState:
    """
    Sample initial position and attitude using the shared reset distribution.

    This avoids mjx.forward during free-flyer policy-training rollouts. It is
    valid because the compact backend only needs qpos, body velocity, angular
    velocity, controls, time, and terminal counters.
    """
    rng_key, qpos, qvel = sample_initial_conditions(
        rng_key, config.init_qpos[0], config.init_qvel[0], config.num_envs,
        config.max_start_offset, config.max_start_linear_velocity,
        config.max_start_angular_velocity)
    # Match MJX's world-frame reset velocity before storing it in body coordinates.
    rot_world_to_body = jnp.swapaxes(quaternion_to_rotation_matrix(qpos[:, 3:7]), 1, 2)
    vel_body = jnp.einsum("bij,bj->bi", rot_world_to_body, qvel[:, :3])
    return FreeFlyerVecEnvState(
        rng=rng_key,
        qpos=qpos,
        vel_body=vel_body,
        omega=qvel[:, 3:6],
        time=jnp.zeros((config.num_envs,), dtype=qpos.dtype),
        ctrl=jnp.zeros(
            (config.num_envs, config.thruster_mixer_T.shape[0]),
            dtype=qpos.dtype,
        ),
        actuator_force=jnp.zeros(
            (config.num_envs, config.thruster_mixer_T.shape[0]),
            dtype=qpos.dtype,
        ),
        terminal_hold_counts=jnp.zeros((config.num_envs,), dtype=jnp.int32),
        disturbance_states=config.base_disturbance_states,
        perturbation_states=config.base_perturbation_states,
    )


def reset_masked(
    state: FreeFlyerVecEnvState,
    config: VecEnvStepConfig,
    reset_mask: jnp.ndarray,
) -> FreeFlyerVecEnvState:
    reset_mask = validate_reset_mask(reset_mask, config.num_envs)
    reset_state = reset_from_key(state.rng, config=config)
    # Effects persist through episode resets, including custom internal state.
    reset_state = reset_state.replace(disturbance_states=state.disturbance_states,
                                      perturbation_states=state.perturbation_states)
    return merge_reset_rows(reset_state, state, reset_mask).replace(rng=reset_state.rng)


def prepare_effects(
    state: FreeFlyerVecEnvState,
    base_ctrl: jnp.ndarray,
) -> Tuple[FreeFlyerVecEnvState, jnp.ndarray, jnp.ndarray]:
    force_world, disturbances = apply_world_wrenches(
        state.disturbance_states, state.time,
        jnp.zeros((base_ctrl.shape[0], 6), dtype=base_ctrl.dtype))
    control, perturbations = apply_actuator_effects(
        state.perturbation_states, base_ctrl, state.time)
    return state.replace(disturbance_states=disturbances,
                         perturbation_states=perturbations), control, force_world


def actuator_forces(control, config):
    """Apply direct-actuator control and force limits once per action."""
    actuator_force = control
    for limits in (getattr(config, 'actuator_control_limits', None),
                   getattr(config, 'actuator_force_limits', None)):
        if limits is not None:
            actuator_force = jnp.clip(actuator_force, limits[:, 0], limits[:, 1])
    return actuator_force


def integrate_substep(
    state: FreeFlyerVecEnvState,
    ctrl: jnp.ndarray,
    qfrc_applied: jnp.ndarray,
    config: VecEnvStepConfig,
    actuator_force: jnp.ndarray | None = None,
) -> FreeFlyerVecEnvState:
    if actuator_force is None:
        actuator_force = actuator_forces(ctrl, config)
    wrench = actuator_force @ config.thruster_mixer_T
    rot_body_to_world = quaternion_to_rotation_matrix(state.qpos[:, 3:7])
    rot_world_to_body = jnp.swapaxes(rot_body_to_world, 1, 2)
    force_body = wrench[:, :3] + jnp.einsum(
        "bij,bj->bi", rot_world_to_body, qfrc_applied[:, :3]
    )
    torque_body = wrench[:, 3:6] + jnp.einsum(
        "bij,bj->bi", rot_world_to_body, qfrc_applied[:, 3:6]
    )
    quat_t = quaternion_rate_matrix(state.qpos[:, 3:7])
    dt = jnp.asarray(config.model_dt, dtype=state.qpos.dtype)
    mass = jnp.asarray(config.mass, dtype=state.qpos.dtype)
    inertia = jnp.asarray(config.inertia_diag, dtype=state.qpos.dtype)

    pos_dot = jnp.einsum("bij,bj->bi", rot_body_to_world, state.vel_body)
    quat_dot = jnp.einsum("bij,bj->bi", quat_t, state.omega)
    vel_dot = force_body / mass - jnp.cross(state.omega, state.vel_body)
    inertia_omega = state.omega * inertia
    omega_dot = (torque_body - jnp.cross(state.omega, inertia_omega)) / inertia

    qpos_next = jnp.concatenate(
        (
            state.qpos[:, :3] + dt * pos_dot,
            state.qpos[:, 3:7] + dt * quat_dot,
        ),
        axis=1,
    )
    quat_next = qpos_next[:, 3:7]
    quat_next = quat_next / (jnp.linalg.norm(quat_next, axis=1, keepdims=True) + 1e-9)
    qpos_next = qpos_next.at[:, 3:7].set(quat_next)
    return state.replace(
        qpos=qpos_next,
        vel_body=state.vel_body + dt * vel_dot,
        omega=state.omega + dt * omega_dot,
        time=state.time + dt,
        ctrl=ctrl,
        actuator_force=actuator_force,
    )


def _advance_and_evaluate(
    state: FreeFlyerVecEnvState,
    commanded_ctrl: jnp.ndarray,
    next_waypoint: jnp.ndarray,
    config: VecEnvStepConfig,
    prev_residuals: Optional[jnp.ndarray] = None,
    prev_states: Optional[jnp.ndarray] = None,
):
    commanded_ctrl = jnp.asarray(commanded_ctrl)

    if prev_states is None:
        prev_states = state_features(state, next_waypoint)
    if config.effects_enabled:
        prepared_state, applied_ctrl, qfrc_applied = prepare_effects(
            state, commanded_ctrl
        )
    else:
        prepared_state = state
        applied_ctrl = commanded_ctrl
        qfrc_applied = jnp.zeros((state.qpos.shape[0], 6), dtype=commanded_ctrl.dtype)

    applied_force = actuator_forces(applied_ctrl, config)
    next_state = jax.lax.fori_loop(
        0,
        config.control_decimation,
        lambda _i, sub_state: integrate_substep(
            sub_state,
            applied_ctrl,
            qfrc_applied,
            config,
            applied_force,
        ),
        prepared_state,
    )
    next_states = state_features(next_state, next_waypoint)

    actual_wrench = jnp.atleast_2d(next_state.actuator_force) @ config.thruster_mixer_T
    termination, reward_result = evaluate_transition(
        prev_states, next_states, commanded_ctrl, state.terminal_hold_counts,
        state.actuator_force, state.ctrl, actual_wrench, config, prev_residuals)
    next_state = next_state.replace(terminal_hold_counts=termination.terminal_hold_counts)
    return next_state, prev_states, next_states, applied_ctrl, actual_wrench, termination, reward_result


def step_with_observations(
    state: FreeFlyerVecEnvState,
    commanded_ctrl: jnp.ndarray,
    next_waypoint: jnp.ndarray,
    config: VecEnvStepConfig,
    prev_residuals: Optional[jnp.ndarray] = None,
) -> Tuple[FreeFlyerVecEnvState, VecEnvStepOutput]:
    """Expose full observations for evaluation using the same transition as training."""
    result = _advance_and_evaluate(state, commanded_ctrl, next_waypoint, config, prev_residuals)
    next_state, previous, following, applied, wrench, termination, reward = result
    return next_state, full_output(previous, following, commanded_ctrl, applied,
        wrench, termination, reward, observations(state),
        observations(next_state), config)


def step(state, actions, reference, config, previous_residual=None, previous_features=None):
    """Advance one control interval; return next state, lean output and next features."""
    next_state, previous, following, applied, wrench, termination, reward = _advance_and_evaluate(
        state, actions, reference, config, previous_residual, previous_features)
    output = training_output(previous, following, applied, wrench, termination, reward, config)
    return next_state, output, following


def reset(state, config, reset_mask=None):
    if reset_mask is None:
        return reset_from_key(state.rng, config=config)
    return reset_masked(state, config, reset_mask)
