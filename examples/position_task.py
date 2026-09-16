"""A position-only task using the existing free-flyer environment and builders."""
from dataclasses import replace
import jax.numpy as jnp

import examples.custom_rewards  # Register the selectable reward functions.
from smallsat_sim.api import register_termination
from smallsat_sim.api.experiments import get_env, register_env
from smallsat_sim.envs.termination import full_pose_termination, TerminationResult


@register_termination('examples/position_only/v1')
def position_termination(context):
    config = context.config
    within = (jnp.linalg.norm(context.next_states[:, :3], axis=-1) < config.terminal_radius)
    within &= jnp.linalg.norm(context.next_states[:, 6:9], axis=-1) < config.terminal_max_speed
    counts = jnp.where(within, context.terminal_hold_counts + 1, 0)
    success = counts >= config.terminal_hold_steps
    failure = full_pose_termination(context).failure_terminals
    return TerminationResult(success | failure, success, failure, counts)


base_entry = get_env('astrobee_rl')


def position_config(spec):
    config = base_entry.config_builder(spec)
    config.env.environment.task = 'position_only'
    return config


register_env('examples/position_only', replace(base_entry, config_builder=position_config))
