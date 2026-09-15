"""Configuration captured once when constructing a numerical step."""
from dataclasses import dataclass
from typing import Optional, Tuple
import jax.numpy as jnp
import numpy as np
from mujoco import mjx
from smallsat_sim.envs.effects.disturbances import DisturbanceState
from smallsat_sim.envs.effects.actuator_kernels import BatchedFaultState

@dataclass(eq=False, frozen=True)
class VecEnvStepConfig:
    """Captured task/physics settings; full resets restore base effects.

    Masked episode resets preserve current effects. Rebuild the collector when
    changing static settings rather than mutating a captured configuration.
    """

    mjx_model: mjx.Model
    init_qpos: jnp.ndarray
    init_qvel: jnp.ndarray
    num_envs: int
    max_start_offset: float
    control_decimation: int
    sigma_pos: float
    sigma_vel: float
    sigma_att: float
    sigma_angvel: float
    w_pos: float
    w_vel: float
    w_att: float
    w_angvel: float
    lam_fuel: float
    lam_speed_terminal: float
    lam_ang_speed_terminal: float
    lam_fuel_terminal: float
    terminal_bonus: float
    terminal_radius: float
    terminal_max_speed: float
    terminal_max_att_error: float
    terminal_max_ang_speed: float
    lam_wrench_residual: float
    wrench_residual_tolerance: float
    wrench_residual_clip: float
    terminal_hold_steps: int
    enable_failure_termination: bool
    failure_max_position_error: float
    failure_max_speed: float
    failure_max_att_error: float
    failure_max_ang_speed: float
    res_dim: int
    use_adaptive_approach: bool
    collect_reward_components: bool
    thruster_mixer_T: jnp.ndarray
    reward: str = "vec_env/full_pose"
    termination: str = "full_pose"
    base_disturbance_states: Tuple[Optional[DisturbanceState], ...] = ()
    base_perturbation_states: Tuple[Optional[BatchedFaultState], ...] = ()
    mass: float = 1.0
    inertia_diag: Optional[jnp.ndarray] = None
    model_dt: float = 0.01
    actuator_control_limits: Optional[jnp.ndarray] = None
    actuator_force_limits: Optional[jnp.ndarray] = None
    effects_enabled: bool = True
    max_start_linear_velocity: float = 0.0
    max_start_angular_velocity: float = 0.0
    # Model arrays with a leading environment axis (explicit scenario batches).
    batched_model_fields: tuple[str, ...] = ()


def build_step_config(env, *, effects_enabled: bool = True) -> VecEnvStepConfig:
    """Capture current task and physics settings."""
    task = env.env_cfg.environment
    return VecEnvStepConfig(
        mjx_model=env.mjx_model,
        init_qpos=env.init_qpos,
        init_qvel=env.init_qvel,
        num_envs=env.num_envs,
        max_start_offset=env.max_start_offset,
        max_start_linear_velocity=env.max_start_linear_velocity,
        max_start_angular_velocity=env.max_start_angular_velocity,
        control_decimation=int(env.env_cfg.environment.control_decimation),
        sigma_pos=float(task.sigma_pos),
        sigma_vel=float(task.sigma_vel),
        sigma_att=float(task.sigma_att),
        sigma_angvel=float(task.sigma_angvel),
        w_pos=float(task.w_pos),
        w_vel=float(task.w_vel),
        w_att=float(task.w_att),
        w_angvel=float(task.w_angvel),
        lam_fuel=float(task.lam_fuel),
        lam_speed_terminal=float(task.lam_speed_terminal),
        lam_ang_speed_terminal=float(task.lam_ang_speed_terminal),
        lam_fuel_terminal=float(task.lam_fuel_terminal),
        terminal_bonus=float(task.terminal_bonus),
        terminal_radius=float(task.terminal_radius),
        terminal_max_speed=float(task.terminal_max_speed),
        terminal_max_att_error=float(task.terminal_max_att_error),
        terminal_max_ang_speed=float(task.terminal_max_ang_speed),
        terminal_hold_steps=int(task.terminal_hold_steps),
        enable_failure_termination=bool(task.enable_failure_termination),
        failure_max_position_error=float(task.failure_max_position_error),
        failure_max_speed=float(task.failure_max_speed),
        failure_max_att_error=float(task.failure_max_att_error),
        failure_max_ang_speed=float(task.failure_max_ang_speed),
        lam_wrench_residual=float(task.lam_wrench_residual),
        wrench_residual_tolerance=float(task.wrench_residual_tolerance),
        wrench_residual_clip=float(task.wrench_residual_clip),
        res_dim=int(env.res_dim),
        use_adaptive_approach=bool(env.use_adaptive_approach),
        collect_reward_components=bool(env.collect_reward_components),
        reward=str(getattr(env.env_cfg.environment, "reward", "vec_env/full_pose")),
        termination=env.env_cfg.environment.termination,
        effects_enabled=bool(effects_enabled),
        thruster_mixer_T=env._thruster_mixer_T,
        mass=float(env.model_cfg.physical.mass),
        inertia_diag=jnp.asarray(env.model_cfg.physical.diag_inertia, dtype=jnp.float32),
        model_dt=float(env.model.opt.timestep),
        actuator_control_limits=_actuator_limits(env.model.actuator_ctrlrange, env.model.actuator_ctrllimited),
        actuator_force_limits=_actuator_limits(env.model.actuator_forcerange, env.model.actuator_forcelimited),
        base_disturbance_states=env.disturbance_states,
        base_perturbation_states=env.perturbation_states,
    )


def _actuator_limits(ranges, limited):
    return jnp.asarray(np.where(limited[:, None], ranges, [-np.inf, np.inf]), dtype=jnp.float32)


def validate_physics(model, backend):
    """Reject models outside the vector task's explicit physics assumptions."""
    if model.nq != 7 or model.nv != 6 or model.nbody != 2:
        raise ValueError("Vector environments require exactly one free body (nq=7, nv=6)")
    if model.opt.timestep <= 0 or model.body_mass[1] <= 0 or np.any(model.body_inertia[1] <= 0):
        raise ValueError("Physics timestep, mass and principal inertias must be positive")
    if backend != "freeflyer":
        return
    if (model.neq or model.ntendon or np.any(model.opt.gravity)
            or np.any(model.dof_damping) or np.any(model.dof_frictionloss)
            or np.any(model.dof_armature) or np.any(model.jnt_stiffness)
            or model.opt.viscosity or model.opt.density):
        raise ValueError("Freeflyer requires unconstrained flight without passive forces; use MJX")
    collision_pairs = (
        (model.geom_contype[:, None] & model.geom_conaffinity[None, :]) != 0
    ) & (model.geom_bodyid[:, None] != model.geom_bodyid[None, :])
    if model.npair or np.any(collision_pairs):
        raise ValueError("Freeflyer does not support contact geometry; use MJX")
    if (not np.allclose(model.body_ipos[1], 0)
            or not np.allclose(np.abs(model.body_iquat[1]), [1, 0, 0, 0])):
        raise ValueError("Freeflyer requires a centered body with diagonal body-frame inertia")
    if (np.any(model.actuator_dyntype) or np.any(model.actuator_gaintype)
            or np.any(model.actuator_biastype)
            or not np.allclose(model.actuator_gainprm[:, 0], 1)):
        raise ValueError("Freeflyer requires direct unit-gain actuators; use MJX")
