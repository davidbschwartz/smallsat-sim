"""Full-pose reward potentials, costs and their explicitly ordered composition."""
import jax.numpy as jnp

from smallsat_sim.api.rewards import RewardContext, RewardResult


def position_potential(states, config):
    sigma = jnp.asarray(config.sigma_pos, dtype=states.dtype)
    weight = jnp.asarray(config.w_pos, dtype=states.dtype)
    return weight * jnp.exp(-jnp.sum((states[:, 0:3] / sigma) ** 2, axis=1))


def velocity_potential(states, config):
    sigma = jnp.asarray(config.sigma_vel, dtype=states.dtype)
    weight = jnp.asarray(config.w_vel, dtype=states.dtype)
    return weight * jnp.exp(-jnp.sum((states[:, 6:9] / sigma) ** 2, axis=1))


def attitude_potential(states, config):
    sigma = jnp.asarray(config.sigma_att, dtype=states.dtype)
    weight = jnp.asarray(config.w_att, dtype=states.dtype)
    norm = jnp.linalg.norm(states[:, 3:6], axis=1)
    return weight * jnp.exp(-((norm / sigma) ** 2))


def angular_velocity_potential(states, config):
    sigma = jnp.asarray(config.sigma_angvel, dtype=states.dtype)
    weight = jnp.asarray(config.w_angvel, dtype=states.dtype)
    return weight * jnp.exp(-jnp.sum((states[:, 9:12] / sigma) ** 2, axis=1))


def fuel_cost(context):
    """Charge requested actuator commands, before faults change realized thrust."""
    weight = jnp.asarray(context.config.lam_fuel, dtype=context.prev_states.dtype)
    return weight * jnp.sum(jnp.abs(context.actions), axis=1)


def terminal_speed_cost(context):
    dtype = context.prev_states.dtype
    weight = jnp.asarray(context.config.lam_speed_terminal, dtype=dtype)
    speed_squared = jnp.sum(context.next_states[:, 6:9] ** 2, axis=1)
    return weight * speed_squared * context.termination.success_terminals.astype(dtype)


def terminal_angular_speed_cost(context):
    dtype = context.prev_states.dtype
    weight = jnp.asarray(context.config.lam_ang_speed_terminal, dtype=dtype)
    speed_squared = jnp.sum(context.next_states[:, 9:12] ** 2, axis=1)
    return weight * speed_squared * context.termination.success_terminals.astype(dtype)


def terminal_fuel_cost(context):
    dtype = context.prev_states.dtype
    weight = jnp.asarray(context.config.lam_fuel_terminal, dtype=dtype)
    fuel = jnp.sum(jnp.abs(context.actions), axis=1)
    return weight * fuel * context.termination.success_terminals.astype(dtype)


def wrench_residual_cost(context):
    config = context.config
    dtype = context.prev_states.dtype
    if not config.use_adaptive_approach or config.res_dim <= 0:
        return jnp.zeros((context.prev_states.shape[0],), dtype=dtype)
    weight = jnp.asarray(config.lam_wrench_residual, dtype=dtype)
    tolerance = jnp.asarray(config.wrench_residual_tolerance, dtype=dtype)
    clip_value = jnp.asarray(config.wrench_residual_clip, dtype=dtype)
    if context.prev_residuals is None or context.prev_residuals.shape[-1] == 0:
        residuals = jnp.zeros((context.prev_states.shape[0], config.res_dim), dtype=dtype)
    else:
        residuals = context.prev_residuals[:, :6]
    excess = jnp.maximum(jnp.linalg.norm(residuals, axis=1) - tolerance, 0.)
    return weight * jnp.minimum(excess, clip_value)


def success_bonus(context):
    dtype = context.prev_states.dtype
    return jnp.asarray(context.config.terminal_bonus, dtype=dtype) * context.termination.success_terminals.astype(dtype)


def full_pose_reward(
    context: RewardContext, *,
    position=position_potential,
    velocity=velocity_potential,
    attitude=attitude_potential,
    angular_velocity=angular_velocity_potential,
    fuel=fuel_cost,
    terminal_speed=terminal_speed_cost,
    terminal_angular_speed=terminal_angular_speed_cost,
    terminal_fuel=terminal_fuel_cost,
    wrench_residual=wrench_residual_cost,
    bonus=success_bonus,
) -> RewardResult:
    """Replace individual functions while retaining default arithmetic and diagnostics.

    Potentials accept (states, config); costs and bonus accept RewardContext.
    Every function returns one scalar per environment. Costs are subtracted.
    """
    config = context.config
    if config is None:
        raise ValueError("full_pose reward requires a VecEnvStepConfig-compatible config.")
    pos_curr = position(context.prev_states, config)
    vel_curr = velocity(context.prev_states, config)
    att_curr = attitude(context.prev_states, config)
    ang_curr = angular_velocity(context.prev_states, config)
    pos_next = position(context.next_states, config)
    vel_next = velocity(context.next_states, config)
    att_next = attitude(context.next_states, config)
    ang_next = angular_velocity(context.next_states, config)
    phi_curr = pos_curr + vel_curr + att_curr + ang_curr
    phi_next = pos_next + vel_next + att_next + ang_next
    fuel_pen = fuel(context)
    speed_pen = terminal_speed(context)
    angular_pen = terminal_angular_speed(context)
    terminal_fuel_pen = terminal_fuel(context)
    residual_pen = wrench_residual(context)
    penalties = fuel_pen + speed_pen + angular_pen + terminal_fuel_pen
    # Preserve the default non-adaptive graph: it never added a zero residual cost.
    if wrench_residual is not wrench_residual_cost or (config.use_adaptive_approach and config.res_dim > 0):
        penalties = penalties + residual_pen
    terminal_reward = bonus(context)
    rewards = phi_next - phi_curr - penalties
    rewards = rewards + terminal_reward
    components = {
        "shaping_pos": pos_next - pos_curr,
        "shaping_vel": vel_next - vel_curr,
        "shaping_att": att_next - att_curr,
        "shaping_angvel": ang_next - ang_curr,
        "shaping_total": phi_next - phi_curr,
        "penalty_fuel": fuel_pen,
        "penalty_terminal_speed": speed_pen,
        "penalty_terminal_ang_speed": angular_pen,
        "penalty_terminal_fuel": terminal_fuel_pen,
        "penalty_wrench_residual": residual_pen,
        "penalty_total": penalties,
        "bonus_terminal": terminal_reward,
        "terminated_success": context.termination.success_terminals.astype(rewards.dtype),
        "terminated_failure": context.termination.failure_terminals.astype(rewards.dtype),
        "reward_total": rewards,
    }
    expected_shape = (context.prev_states.shape[0],)
    for name, value in components.items():
        if getattr(value, "shape", None) != expected_shape:
            raise ValueError(f"Reward component {name!r} must have shape {expected_shape}")
    return RewardResult(rewards, components if config.collect_reward_components else {"reward_total": rewards})
