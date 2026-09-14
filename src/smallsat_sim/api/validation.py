"""Validation helpers for public API specifications."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from smallsat_sim.api.rewards import RewardContext, RewardResult
from smallsat_sim.envs.termination import TerminationResult
from smallsat_sim.model.vehicle import validate_vehicle
from smallsat_sim.errors import ValidationError


def validate_reward(
    reward_fn,
    *,
    num_envs: int = 2,
    state_dim: int = 12,
    action_dim: int = 4,
    config: Any | None = None,
) -> RewardResult:
    context = RewardContext(
        prev_states=jnp.zeros((num_envs, state_dim), dtype=jnp.float32),
        next_states=jnp.zeros((num_envs, state_dim), dtype=jnp.float32),
        actions=jnp.zeros((num_envs, action_dim), dtype=jnp.float32),
        config=config,
        termination=TerminationResult(
            jnp.zeros((num_envs,), dtype=bool),
            jnp.zeros((num_envs,), dtype=bool),
            jnp.zeros((num_envs,), dtype=bool),
            jnp.zeros((num_envs,), dtype=jnp.int32),
        ),
    )
    result = reward_fn(context)
    if not isinstance(result, RewardResult):
        raise ValidationError("Reward function must return RewardResult.")
    expected = (num_envs,)
    for field_name in (
        "rewards",
    ):
        value = getattr(result, field_name)
        if value.shape != expected:
            raise ValidationError(
                f"RewardResult.{field_name} must have shape {expected}, got {value.shape}."
            )
    return result
