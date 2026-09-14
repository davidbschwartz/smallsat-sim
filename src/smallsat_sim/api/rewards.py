"""Reward specification helpers and registry bindings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

import jax.numpy as jnp
from smallsat_sim.envs.termination import TerminationResult


@dataclass(frozen=True)
class RewardResult:
    """Reward values and diagnostics; episode endings belong to termination."""

    rewards: jnp.ndarray
    components: Mapping[str, jnp.ndarray]


@dataclass(frozen=True)
class RewardContext:
    """Transition features and the already-computed termination decision."""

    prev_states: jnp.ndarray
    next_states: jnp.ndarray
    actions: jnp.ndarray
    config: Any
    termination: TerminationResult
    actual_wrench: jnp.ndarray | None = None
    desired_wrench: jnp.ndarray | None = None
    prev_residuals: jnp.ndarray | None = None


class RewardCallable(Protocol):
    def __call__(self, context: RewardContext) -> RewardResult: ...


RewardFunction = RewardCallable


@dataclass(frozen=True)
class RewardTerm:
    """One batched scalar function and its signed weight."""
    func: Callable[[RewardContext], jnp.ndarray]
    weight: float = 1.0


def compose_reward(terms: Mapping[str, RewardTerm | Callable[[RewardContext], jnp.ndarray]]) -> RewardCallable:
    """Build a reward from named functions; positive weights add, negative subtract.

    The mapping is copied at construction. Zero-weight terms are not evaluated.
    Register the returned function under a stable name to select it in a task.
    """
    import math

    selected = []
    for name, term in terms.items():
        if not isinstance(name, str) or not name.strip() or name == "reward_total":
            raise ValueError("Reward term names must be nonempty and cannot be 'reward_total'")
        term = term if isinstance(term, RewardTerm) else RewardTerm(term)
        if not callable(term.func) or not math.isfinite(term.weight):
            raise ValueError(f"Reward term {name!r} requires a callable and a finite weight")
        if term.weight != 0:
            selected.append((name, term))
    selected = tuple(selected)

    def reward(context: RewardContext) -> RewardResult:
        total = jnp.zeros((context.prev_states.shape[0],), dtype=context.prev_states.dtype)
        components = {}
        for name, term in selected:
            value = jnp.asarray(term.func(context), dtype=total.dtype)
            if value.shape != total.shape:
                raise ValueError(f"Reward term {name!r} must return shape {total.shape}, got {value.shape}")
            weighted = jnp.asarray(term.weight, dtype=total.dtype) * value
            total = total + weighted
            components[name] = weighted
        components["reward_total"] = total
        return RewardResult(total, components if getattr(context.config, "collect_reward_components", True)
                            else {"reward_total": total})

    return reward
