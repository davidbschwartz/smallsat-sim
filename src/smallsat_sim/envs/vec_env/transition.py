"""Task evaluation and output formatting shared by both physics backends."""
import jax.numpy as jnp
from smallsat_sim.api.registry import get_reward, get_termination
from smallsat_sim.api import vec_env_rewards as _registered_rewards
from smallsat_sim.api.rewards import RewardContext
from smallsat_sim.envs.termination import TerminationContext
from .types import VecEnvStepOutput, VecEnvTrainingStepOutput


def evaluate_transition(previous_features, next_features, actions, previous_counts,
                        previous_force, previous_control, actual_wrench, config,
                        previous_residual=None):
    """Evaluate task endings before reward; residuals refer to the previous interval."""
    termination = get_termination(config.termination)(
        TerminationContext(next_features, previous_counts, config))
    if config.use_adaptive_approach and config.res_dim > 0:
        if previous_residual is None or previous_residual.shape[-1] == 0:
            previous_residual = (
                jnp.atleast_2d(previous_force) @ config.thruster_mixer_T
                - jnp.atleast_2d(previous_control) @ config.thruster_mixer_T)
        previous_residual = jnp.asarray(previous_residual, dtype=previous_features.dtype)
    else:
        previous_residual = None
    reward = get_reward(config.reward)(RewardContext(
        prev_states=previous_features, next_states=next_features, actions=actions,
        config=config, termination=termination, actual_wrench=actual_wrench,
        desired_wrench=actions @ config.thruster_mixer_T,
        prev_residuals=previous_residual))
    return termination, reward


def training_output(previous_features, next_features, applied_control, actual_wrench,
                    termination, reward, config):
    """Keep observations and optional diagnostics out of the training scan."""
    return VecEnvTrainingStepOutput(
        prev_states=previous_features,
        next_position_error=next_features[:, :3],
        next_attitude_error=jnp.linalg.norm(next_features[:, 3:6], axis=1),
        next_speed=jnp.linalg.norm(next_features[:, 6:9], axis=1),
        next_angular_speed=jnp.linalg.norm(next_features[:, 9:12], axis=1),
        rewards=reward.rewards, terminals=termination.terminals,
        applied_ctrl=applied_control, actual_wrench=actual_wrench,
        success_terminals=termination.success_terminals,
        failure_terminals=termination.failure_terminals,
        reward_components=reward.components if config.collect_reward_components else {})


def full_output(previous_features, next_features, actions, applied_control,
                actual_wrench, termination, reward, previous_obs, next_obs, config):
    return VecEnvStepOutput(
        prev_states=previous_features, next_states=next_features,
        rewards=reward.rewards, terminals=termination.terminals,
        commanded_ctrl=actions, applied_ctrl=applied_control, actual_wrench=actual_wrench,
        desired_wrench=actions @ config.thruster_mixer_T,
        prev_obs=previous_obs, next_obs=next_obs,
        success_terminals=termination.success_terminals,
        failure_terminals=termination.failure_terminals,
        reward_components=reward.components)
