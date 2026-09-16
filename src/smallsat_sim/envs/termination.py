"""Episode ending decisions and their persistent hold counters."""

from dataclasses import dataclass
from typing import Any, Protocol

import jax.numpy as jnp


@dataclass(frozen=True)
class TerminationContext:
    next_states: jnp.ndarray
    terminal_hold_counts: jnp.ndarray
    config: Any


@dataclass(frozen=True)
class TerminationResult:
    terminals: jnp.ndarray
    success_terminals: jnp.ndarray
    failure_terminals: jnp.ndarray
    terminal_hold_counts: jnp.ndarray


class TerminationCallable(Protocol):
    def __call__(self, context: TerminationContext) -> TerminationResult: ...


def full_pose_termination(context: TerminationContext) -> TerminationResult:
    success, counts = _compute_terminals(
        context.next_states, context.terminal_hold_counts, context.config,
    )
    failure = _compute_failures(context.next_states, context.config)
    return TerminationResult(jnp.logical_or(success, failure), success, failure, counts)


def _compute_terminals(
    states: jnp.ndarray,
    terminal_hold_counts: jnp.ndarray,
    config: Any,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    radius = jnp.asarray(config.terminal_radius, dtype=states.dtype)
    max_att_error = jnp.asarray(config.terminal_max_att_error, dtype=states.dtype)
    required_steps = max(int(config.terminal_hold_steps), 1)
    pos_ok = jnp.linalg.norm(states[:, 0:3], axis=1) <= radius
    att_ok = jnp.linalg.norm(states[:, 3:6], axis=1) <= max_att_error
    in_terminal_set = jnp.logical_and(pos_ok, att_ok)
    counts_next = jnp.where(in_terminal_set, terminal_hold_counts + 1, 0)
    terminals = counts_next >= required_steps
    return terminals, counts_next


def _compute_failures(states: jnp.ndarray, config: Any) -> jnp.ndarray:
    if not bool(config.enable_failure_termination):
        return jnp.zeros((states.shape[0],), dtype=bool)

    max_pos = jnp.asarray(config.failure_max_position_error, dtype=states.dtype)
    max_speed = jnp.asarray(config.failure_max_speed, dtype=states.dtype)
    max_att = jnp.asarray(config.failure_max_att_error, dtype=states.dtype)
    max_ang = jnp.asarray(config.failure_max_ang_speed, dtype=states.dtype)

    pos_fail = jnp.linalg.norm(states[:, 0:3], axis=1) > max_pos
    speed_fail = jnp.linalg.norm(states[:, 6:9], axis=1) > max_speed
    att_fail = jnp.linalg.norm(states[:, 3:6], axis=1) > max_att
    ang_fail = jnp.linalg.norm(states[:, 9:12], axis=1) > max_ang
    return jnp.logical_or(
        pos_fail,
        jnp.logical_or(speed_fail, jnp.logical_or(att_fail, ang_fail)),
    )


