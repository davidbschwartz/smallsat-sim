"""Register a modified full-pose reward and a reward composed from scratch."""
import jax.numpy as jnp

from smallsat_sim.api import RewardTerm, compose_reward, register_reward
from smallsat_sim.envs.rewards import full_pose_reward


def command_effort(context):
    return jnp.sum(context.actions ** 2, axis=-1)


def quadratic_fuel_cost(context):
    return context.config.lam_fuel * command_effort(context)


@register_reward('examples/quadratic_fuel/v1')
def quadratic_fuel_reward(context):
    return full_pose_reward(context, fuel=quadratic_fuel_cost)


def position_error(context):
    return jnp.linalg.norm(context.next_states[:, :3], axis=-1)


register_reward('examples/position_effort/v1', compose_reward({
    'position': RewardTerm(position_error, weight=-1.),
    'effort': RewardTerm(command_effort, weight=-.02),
}))
