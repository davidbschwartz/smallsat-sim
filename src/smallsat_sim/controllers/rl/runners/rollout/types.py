"""Typed rollout state containers and callback signatures."""

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp

from .context import ContextHistory

PolicyInputFn = Callable[
    [int, jnp.ndarray, jnp.ndarray, Any],
    tuple[jnp.ndarray, Any],
]
SamplePolicyFn = Callable[
    [int, jnp.ndarray, jnp.ndarray, Any],
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Any],
]
UpdateContextFn = Callable[
    [int, Any, jnp.ndarray, jnp.ndarray, jnp.ndarray, Any],
    tuple[jnp.ndarray, Any, Any],
]
BootstrapValueFn = Callable[
    [int, Any, jnp.ndarray, jnp.ndarray, Any],
    tuple[jnp.ndarray, jnp.ndarray, Any],
]
StateFeaturesFn = Callable[[Any, jnp.ndarray], jnp.ndarray]
# step(state, actions, reference, config, context, previous_features)
# returns (next_state, output, next_features); reset always accepts a per-env mask.
StepFn = Callable[
    [Any, jnp.ndarray, jnp.ndarray, Any, jnp.ndarray, jnp.ndarray],
    tuple[Any, Any, jnp.ndarray],
]
ResetFn = Callable[[Any, Any, jnp.ndarray], Any]


class PolicyState(NamedTuple):
    """Policy-side arrays carried through collection, separate from physics."""
    history: ContextHistory
    latent_actions: jax.Array


class TransitionLabels(NamedTuple):
    """Learning targets recorded after a transition, before history resets."""
    latent_actions: jax.Array
    normalized_context: jax.Array
    history_full: jax.Array


class RolloutCarry(NamedTuple):
    """Only these values advance from one scan iteration to the next."""

    env_state: Any
    features: jax.Array
    context: jax.Array
    key: jax.Array
    policy_state: Any
    episode_return: jax.Array
    episode_length: jax.Array


@dataclass
class FunctionalRolloutCallbacks:
    """
    Collection of callables used by `run_functional_rollout` to interact with
    policy logic outside the environment stepping loop.

    Each callable receives the current step index together with the evolving
    carry so callers can maintain additional per-rollout state (e.g. history
    buffers for the adaptation module).
    """

    prepare_policy_input: PolicyInputFn
    sample_policy: SamplePolicyFn
    update_context: UpdateContextFn
    bootstrap_value: BootstrapValueFn


@dataclass
class FunctionalRolloutResult:
    """
    Collected arrays have leading axes [time, environment].

    done_masks marks both terminals and timeouts; bootstrap_values contains
    pre-reset values only at timeouts and the final rollout step.
    """

    step_outputs: Any
    actions: jnp.ndarray
    values: jnp.ndarray
    logp: jnp.ndarray
    context: jnp.ndarray
    episode_returns: jnp.ndarray
    done_flags: jnp.ndarray
    done_masks: jnp.ndarray
    terminated_masks: jnp.ndarray
    truncated_masks: jnp.ndarray
    bootstrap_values: jnp.ndarray
    labels: Any
    final_state: Any
    final_context: jnp.ndarray
    final_rng: jnp.ndarray
    final_policy_state: Any
    trajectory: Any = None


@dataclass
class _FunctionalRolloutStep:
    step_output: Any
    actions: jnp.ndarray
    values: jnp.ndarray
    logp: jnp.ndarray
    context: jnp.ndarray
    episode_return: jnp.ndarray
    done_flag: jnp.ndarray
    done_mask: jnp.ndarray
    terminated_mask: jnp.ndarray
    truncated_mask: jnp.ndarray
    bootstrap_value: jnp.ndarray
    labels: Any
    trajectory: Any = None


jax.tree_util.register_dataclass(_FunctionalRolloutStep)
jax.tree_util.register_dataclass(FunctionalRolloutResult)
