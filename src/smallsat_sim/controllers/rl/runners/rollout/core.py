"""Compiled rollout scan over vectorized environment transitions."""

from typing import Any

import jax
import jax.numpy as jnp

from .types import (
    RolloutCarry,
    ResetFn,
    FunctionalRolloutCallbacks,
    FunctionalRolloutResult,
    _FunctionalRolloutStep,
    StateFeaturesFn,
    StepFn,
)


def run_functional_rollout(
    *,
    step_config: Any,
    max_episode_len: int,
    initial_state: Any,
    initial_context: jnp.ndarray,
    rng: jnp.ndarray,
    num_steps: int,
    reference_waypoint: jnp.ndarray,
    callbacks: FunctionalRolloutCallbacks,
    policy_state: Any = None,
    state_features_fn: StateFeaturesFn,
    step_fn: StepFn,
    reset_fn: ResetFn,
    visualization: Any = None,
    single_episode: bool = False,
    record_state_fn: Any = None,
) -> FunctionalRolloutResult:
    """Collect a fixed horizon with independent episode boundaries per environment.

    Policy callbacks own context/history; this scan owns action intervals, episode boundaries,
    timeout bootstrap and reset ordering. Arrays are [environment, features]
    inside the scan and [time, environment, features] in the result.
    single_episode disables resets; evaluation reductions keep only the first
    terminal event in each lane. Optional trajectory recording is evaluation-only.
    """

    # Every backend accepts the full state and supports independently masked resets.
    def _state_features(state):
        return state_features_fn(state, reference_waypoint)

    num_envs = initial_context.shape[0]

    def _scan_body(carry, step_idx: int):
        env_state = carry.env_state
        current_features = carry.features
        context = carry.context
        rng_key = carry.key
        policy_state = carry.policy_state
        episode_return_so_far = carry.episode_return
        episode_length = carry.episode_length

        # Observe and choose an action using the context available before this step.
        policy_input, policy_state = callbacks.prepare_policy_input(
            step_idx, current_features, context, policy_state
        )
        actions, values, logp, rng_key, policy_state = callbacks.sample_policy(
            step_idx, policy_input, rng_key, policy_state
        )

        next_env_state, step_output, next_features = step_fn(
            env_state,
            jnp.nan_to_num(actions) if single_episode else actions,
            reference_waypoint,
            step_config,
            context,
            current_features,
        )

        # Track episode boundaries independently for each environment.
        accumulated_return = episode_return_so_far + step_output.rewards
        accumulated_length = episode_length + 1

        terminated_mask = step_output.terminals.astype(bool)
        timeout_mask = accumulated_length >= max_episode_len
        truncated_mask = jnp.logical_and(timeout_mask, jnp.logical_not(terminated_mask))
        done_mask = jnp.logical_or(terminated_mask, truncated_mask)
        if visualization is not None:
            visualization.observe(step_idx, next_env_state, done_mask)

        last_step = jnp.equal(step_idx, num_steps - 1)
        done_flag = jnp.any(done_mask)
        reset_mask = (
            jnp.zeros_like(done_mask)
            if single_episode
            else jnp.logical_and(done_mask, jnp.logical_not(last_step))
        )

        # A completed transition supplies the next context and adaptation label.
        next_context, transition_labels, policy_state = callbacks.update_context(
            step_idx, step_output, actions, context, reset_mask, policy_state
        )

        # Timeouts and rollout boundaries bootstrap from the state BEFORE reset.
        bootstrap_mask = jnp.logical_or(
            truncated_mask,
            jnp.logical_and(last_step, jnp.logical_not(terminated_mask)),
        )

        def _skip_bootstrap(operand):
            _, _, key, current_policy_state = operand
            return jnp.zeros_like(step_output.rewards), key, current_policy_state

        bootstrap_values_raw, rng_key, policy_state = jax.lax.cond(
            jnp.any(bootstrap_mask),
            lambda args: callbacks.bootstrap_value(step_idx, *args),
            _skip_bootstrap,
            operand=(next_env_state, next_context, rng_key, policy_state),
        )
        bootstrap_values = jnp.where(
            bootstrap_mask,
            bootstrap_values_raw,
            jnp.zeros_like(bootstrap_values_raw),
        )

        episode_return = jnp.where(
            done_mask,
            accumulated_return,
            jnp.zeros_like(accumulated_return),
        )

        # Only finished environments reset; the last step leaves state intact.
        def _reset_after_done(mask):
            reset_state = reset_fn(next_env_state, step_config, mask)
            reset_states = _state_features(reset_state)
            states_after_reset = jnp.where(mask[:, None], reset_states, next_features)
            context_mask = mask[:, None]
            reset_context = jnp.where(
                context_mask,
                jnp.zeros_like(next_context),
                next_context,
            )
            reset_returns = jnp.where(mask, jnp.zeros_like(accumulated_return), accumulated_return)
            reset_lengths = jnp.where(mask, jnp.zeros_like(accumulated_length), accumulated_length)
            return (
                reset_state,
                states_after_reset,
                reset_context,
                reset_returns,
                reset_lengths,
            )

        (
            state_after_reset,
            features_after_reset,
            context_after_reset,
            return_after_reset,
            length_after_reset,
        ) = (
            (next_env_state, next_features, next_context, accumulated_return, accumulated_length)
            if single_episode
            else jax.lax.cond(
                jnp.any(reset_mask),
                _reset_after_done,
                lambda _: (
                    next_env_state,
                    next_features,
                    next_context,
                    accumulated_return,
                    accumulated_length,
                ),
                operand=reset_mask,
            )
        )

        step_record = _FunctionalRolloutStep(
            step_output=step_output,
            actions=actions,
            values=values,
            logp=logp,
            context=context,
            episode_return=episode_return,
            done_flag=done_flag,
            done_mask=done_mask,
            terminated_mask=terminated_mask,
            truncated_mask=truncated_mask,
            bootstrap_value=bootstrap_values,
            labels=transition_labels,
            trajectory=record_state_fn(next_env_state) if record_state_fn else None,
        )

        new_carry = RolloutCarry(
            env_state=state_after_reset,
            features=features_after_reset,
            context=context_after_reset,
            key=rng_key,
            policy_state=policy_state,
            episode_return=return_after_reset,
            episode_length=length_after_reset,
        )
        return new_carry, step_record

    initial_carry = RolloutCarry(
        env_state=initial_state,
        features=_state_features(initial_state),
        context=initial_context,
        key=rng,
        policy_state=policy_state,
        episode_return=jnp.zeros((num_envs,), dtype=jnp.float32),
        episode_length=jnp.zeros((num_envs,), dtype=jnp.int32),
    )
    final_carry, steps = jax.lax.scan(
        _scan_body,
        initial_carry,
        jnp.arange(num_steps, dtype=jnp.int32),
    )

    return FunctionalRolloutResult(
        step_outputs=steps.step_output,
        actions=steps.actions,
        values=steps.values,
        logp=steps.logp,
        context=steps.context,
        episode_returns=steps.episode_return,
        done_flags=steps.done_flag,
        done_masks=steps.done_mask,
        terminated_masks=steps.terminated_mask,
        truncated_masks=steps.truncated_mask,
        bootstrap_values=steps.bootstrap_value,
        labels=steps.labels,
        final_state=final_carry.env_state,
        final_context=final_carry.context,
        final_rng=final_carry.key,
        final_policy_state=final_carry.policy_state,
        trajectory=steps.trajectory,
    )
