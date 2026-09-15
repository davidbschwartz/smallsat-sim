"""MJX physics, explicit-state transitions and episode resets."""
from smallsat_sim.envs.effects.wrenches import apply_world_wrenches
from .reset import sample_initial_conditions, merge_reset_rows, validate_reset_mask
from .transition import evaluate_transition, training_output, full_output

from functools import partial
from typing import Optional, Tuple
import jax
import jax.numpy as jnp
from mujoco import mjx
from smallsat_sim.envs.effects.catalog import apply_actuator_effects
from smallsat_sim.envs.vec_env.observations import mjx_observations, mjx_state_features
from .config import VecEnvStepConfig
from smallsat_sim.envs.vec_env.types import VecEnvState, VecEnvStepOutput


@partial(jax.jit, static_argnames=("num_envs", "max_start_linear_velocity", "max_start_angular_velocity"))
def reset_from_key(
    rng_key: jnp.ndarray,
    *,
    mjx_model: mjx.Model,
    init_qpos: jnp.ndarray,
    init_qvel: jnp.ndarray,
    num_envs: int,
    max_start_offset: float,
    max_start_linear_velocity: float = 0.0,
    max_start_angular_velocity: float = 0.0,
) -> VecEnvState:
    """Sample initial physics from clean model data."""
    rng_key, qpos, qvel = sample_initial_conditions(
        rng_key, init_qpos[0], init_qvel[0], num_envs, max_start_offset,
        max_start_linear_velocity, max_start_angular_velocity)
    # A caller's live batch must never become the reset template.
    clean_data = mjx.make_data(mjx_model)
    batch = jax.vmap(lambda position, velocity: clean_data.replace(
        qpos=position, qvel=velocity))(qpos, qvel)
    batch = jax.vmap(mjx.forward, in_axes=(None, 0))(mjx_model, batch)

    return VecEnvState(
        rng=rng_key,
        mjx_batch=batch,
        terminal_hold_counts=jnp.zeros((num_envs,), dtype=jnp.int32),
    )


def prepare_effects(
    state: VecEnvState,
    base_ctrl: jnp.ndarray,
) -> Tuple[VecEnvState, jnp.ndarray, jnp.ndarray]:
    """
    Pure helper that applies disturbances and perturbations to produce the control
    and generalized forces for the next MuJoCo step.
    """
    force_world, disturbances = apply_world_wrenches(
        state.disturbance_states, state.mjx_batch.time,
        jnp.zeros((base_ctrl.shape[0], 6), dtype=base_ctrl.dtype))
    control, perturbations = apply_actuator_effects(
        state.perturbation_states, base_ctrl, state.mjx_batch.time)
    # Free-joint generalized torque is body-frame; translational force is world-frame.
    torque_body = jnp.einsum("bji,bj->bi", state.mjx_batch.xmat[:, 1], force_world[:, 3:])
    generalized_force = force_world.at[:, 3:].set(torque_body)
    return state.replace(disturbance_states=disturbances,
                         perturbation_states=perturbations), control, generalized_force


def _advance_and_evaluate(state, commanded_ctrl, next_waypoint, config,
                          prev_residuals=None, prev_states=None):
    """Advance physics once, then evaluate termination and reward for either output API."""
    commanded_ctrl = jnp.asarray(commanded_ctrl)

    if prev_states is None:
        prev_states = mjx_state_features(state.mjx_batch, next_waypoint)
    if config.effects_enabled:
        prepared_state, applied_ctrl, qfrc_applied = prepare_effects(
            state, commanded_ctrl
        )
    else:
        prepared_state = state
        applied_ctrl = commanded_ctrl
        qfrc_applied = jnp.zeros_like(state.mjx_batch.qfrc_applied)

    mjx_batch = prepared_state.mjx_batch.replace(
        ctrl=applied_ctrl,
        qfrc_applied=qfrc_applied,
    )

    mjx_batch = advance_physics(config.mjx_model, mjx_batch, config.control_decimation,
                                config.batched_model_fields)

    next_state = prepared_state.replace(mjx_batch=mjx_batch)
    next_states = mjx_state_features(mjx_batch, next_waypoint)

    actual_wrench = jnp.atleast_2d(mjx_batch.actuator_force) @ config.thruster_mixer_T
    termination, reward_result = evaluate_transition(
        prev_states, next_states, commanded_ctrl, state.terminal_hold_counts,
        state.mjx_batch.actuator_force, state.mjx_batch.ctrl, actual_wrench, config,
        prev_residuals)
    next_state = next_state.replace(terminal_hold_counts=termination.terminal_hold_counts)
    return next_state, prev_states, next_states, applied_ctrl, actual_wrench, termination, reward_result


def step_with_observations(
    state: VecEnvState,
    commanded_ctrl: jnp.ndarray,
    next_waypoint: jnp.ndarray,
    config: VecEnvStepConfig,
    prev_residuals: Optional[jnp.ndarray] = None,
) -> Tuple[VecEnvState, VecEnvStepOutput]:
    """
    Functional rollout helper that mirrors VecEnv.transition without touching object
    attributes. Returns the updated VecEnvState together with the per-step transition
    data required for training.
    """
    commanded_ctrl = jnp.asarray(commanded_ctrl)
    (next_state, prev_states, next_states, applied_ctrl, actual_wrench,
     termination, reward_result) = _advance_and_evaluate(
        state, commanded_ctrl, next_waypoint, config, prev_residuals,
    )
    prev_obs = mjx_observations(state.mjx_batch)
    next_obs = mjx_observations(next_state.mjx_batch)

    step_output = full_output(prev_states, next_states, commanded_ctrl, applied_ctrl,
        actual_wrench, termination, reward_result, prev_obs, next_obs, config)

    return next_state, step_output


def reset(state, config, reset_mask=None):
    """Reset episode physics; masked resets retain the realized effects."""
    initial = reset_from_key(
        state.rng, mjx_model=config.mjx_model, init_qpos=config.init_qpos,
        init_qvel=config.init_qvel, num_envs=config.num_envs,
        max_start_offset=config.max_start_offset,
        max_start_linear_velocity=config.max_start_linear_velocity,
        max_start_angular_velocity=config.max_start_angular_velocity)
    if reset_mask is None:
        return initial.replace(disturbance_states=config.base_disturbance_states,
                               perturbation_states=config.base_perturbation_states)
    mask = validate_reset_mask(reset_mask, config.num_envs)
    return state.replace(
        rng=initial.rng,
        mjx_batch=merge_reset_rows(initial.mjx_batch, state.mjx_batch, mask),
        terminal_hold_counts=jnp.where(mask, 0, state.terminal_hold_counts))


def state_features(state, reference):
    return mjx_state_features(state.mjx_batch, reference)


def step(state, actions, reference, config, previous_residual=None, previous_features=None):
    """Advance one control interval; return next state, lean output and next features."""
    next_state, previous, following, applied, wrench, termination, reward = _advance_and_evaluate(
        state, actions, reference, config, previous_residual, previous_features)
    output = training_output(previous, following, applied, wrench, termination, reward, config)
    return next_state, output, following


@partial(jax.jit, static_argnames=("control_decimation", "batched_model_fields"))
def advance_physics(model, batch, control_decimation, batched_model_fields=()):
    """Integrate one held world wrench and actuator command across physics substeps."""
    axes = (jax.tree.map(lambda _: None, model).replace(
        **dict.fromkeys(batched_model_fields, 0)) if batched_model_fields else None)
    advance = jax.vmap(mjx.step, in_axes=(axes, 0))
    torque_world = jnp.einsum("bij,bj->bi", batch.xmat[:, 1], batch.qfrc_applied[:, 3:])
    def substep(_, current):
        torque_body = jnp.einsum("bji,bj->bi", current.xmat[:, 1], torque_world)
        current = current.replace(qfrc_applied=current.qfrc_applied.at[:, 3:].set(torque_body))
        return advance(model, current)
    return jax.lax.fori_loop(0, control_decimation, substep, batch)
