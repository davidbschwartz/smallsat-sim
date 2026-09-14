"""Functional rollout callback and reset behavior tests."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from smallsat_sim.controllers.rl.runners import rollout as ru


@dataclass
class DummyBatch:
    qpos: jnp.ndarray

    def replace(self, **updates: jnp.ndarray) -> "DummyBatch":
        return DummyBatch(qpos=updates.get("qpos", self.qpos))


jax.tree_util.register_dataclass(DummyBatch)


@dataclass
class DummyState:
    rng: jnp.ndarray
    mjx_batch: DummyBatch

    def replace(self, **updates):
        return DummyState(
            rng=updates.get("rng", self.rng),
            mjx_batch=updates.get("mjx_batch", self.mjx_batch),
        )


jax.tree_util.register_dataclass(DummyState)


@dataclass
class DummyStepConfig:
    max_episode_len: int


@dataclass
class DummyStepOutput:
    prev_states: jnp.ndarray
    next_states: jnp.ndarray
    rewards: jnp.ndarray
    terminals: jnp.ndarray
    commanded_ctrl: jnp.ndarray
    applied_ctrl: jnp.ndarray
    actual_wrench: jnp.ndarray
    desired_wrench: jnp.ndarray
    prev_obs: jnp.ndarray
    next_obs: jnp.ndarray
    success_terminals: jnp.ndarray
    failure_terminals: jnp.ndarray
    reward_components: dict


jax.tree_util.register_dataclass(DummyStepOutput)


def test_run_functional_rollout_resets_and_reports_returns() -> None:
    num_envs = 2
    num_steps = 3

    initial_batch = DummyBatch(qpos=jnp.zeros((num_envs, 1), dtype=jnp.float32))
    initial_state = DummyState(
        rng=jax.random.PRNGKey(0),
        mjx_batch=initial_batch,
    )
    step_config = DummyStepConfig(max_episode_len=10)
    initial_context = jnp.zeros((num_envs, 0), dtype=jnp.float32)
    reference_waypoint = jnp.zeros((3,), dtype=jnp.float32)

    def _vecenv_step_stub(state, actions, _waypoint, _config, _context, _features):
        qpos = state.mjx_batch.qpos
        terminals = qpos[:, 0] == 0.0
        next_batch = state.mjx_batch.replace(qpos=qpos + 1.0)
        next_state = state.replace(mjx_batch=next_batch)
        rewards = jnp.full((num_envs,), 2.0, dtype=jnp.float32)
        step_output = DummyStepOutput(
            prev_states=qpos,
            next_states=qpos + 1.0,
            rewards=rewards,
            terminals=terminals,
            commanded_ctrl=actions,
            applied_ctrl=actions,
            actual_wrench=jnp.zeros((num_envs, 1), dtype=jnp.float32),
            desired_wrench=jnp.zeros((num_envs, 1), dtype=jnp.float32),
            prev_obs=qpos,
            next_obs=qpos + 1.0,
            success_terminals=terminals,
            failure_terminals=jnp.zeros_like(terminals),
            reward_components={},
        )
        return next_state, step_output, step_output.next_states

    def _vecenv_reset_stub(state, _config, mask):
        reset_qpos = jnp.full_like(state.mjx_batch.qpos, -1.0)
        return state.replace(
            mjx_batch=state.mjx_batch.replace(
                qpos=jnp.where(mask[:, None], reset_qpos, state.mjx_batch.qpos)
            )
        )

    def _prepare(_step, states, context, policy_state):
        del context
        return states, policy_state

    def _sample(_step, policy_input, rng_key, policy_state):
        actions = jnp.zeros((num_envs, 1), dtype=jnp.float32)
        values = jnp.ones((num_envs,), dtype=jnp.float32)
        logp = jnp.zeros((num_envs,), dtype=jnp.float32)
        return actions, values, logp, rng_key, policy_state

    def _post(_step, step_output, actions, context, reset_flag, policy_state):
        del step_output, actions, reset_flag
        return context, None, policy_state

    def _bootstrap(_step, env_state, context, rng_key, policy_state):
        del env_state, context
        return jnp.zeros((num_envs,), dtype=jnp.float32), rng_key, policy_state

    result = ru.run_functional_rollout(
        step_config=step_config,
        max_episode_len=step_config.max_episode_len,
        initial_state=initial_state,
        initial_context=initial_context,
        rng=initial_state.rng,
        num_steps=num_steps,
        reference_waypoint=reference_waypoint,
        callbacks=ru.FunctionalRolloutCallbacks(
            prepare_policy_input=_prepare,
            sample_policy=_sample,
            update_context=_post,
            bootstrap_value=_bootstrap,
        ),
        state_features_fn=lambda state, _: state.mjx_batch.qpos,
        step_fn=_vecenv_step_stub,
        reset_fn=_vecenv_reset_stub,
    )

    assert bool(result.done_flags[0])
    assert bool(result.done_masks[0].all())
    assert bool(result.terminated_masks[0].all())
    assert not bool(result.truncated_masks[0].any())
    assert jnp.allclose(result.episode_returns[0], jnp.full((num_envs,), 2.0))
    assert jnp.allclose(result.episode_returns[1], jnp.zeros((num_envs,)))
    assert jnp.allclose(result.episode_returns[2], jnp.full((num_envs,), 4.0))
    assert jnp.allclose(result.final_state.mjx_batch.qpos, jnp.ones((num_envs, 1)))


def test_run_functional_rollout_bootstraps_timeouts() -> None:
    num_envs = 2
    num_steps = 5
    max_episode_len = 2

    initial_batch = DummyBatch(qpos=jnp.zeros((num_envs, 1), dtype=jnp.float32))
    initial_state = DummyState(
        rng=jax.random.PRNGKey(0),
        mjx_batch=initial_batch,
    )
    step_config = DummyStepConfig(max_episode_len=max_episode_len)
    initial_context = jnp.zeros((num_envs, 0), dtype=jnp.float32)
    reference_waypoint = jnp.zeros((3,), dtype=jnp.float32)

    def _vecenv_step_stub(state, actions, _waypoint, _config, _context, _features):
        qpos = state.mjx_batch.qpos
        terminals = jnp.zeros((num_envs,), dtype=bool)
        next_batch = state.mjx_batch.replace(qpos=qpos + 1.0)
        next_state = state.replace(mjx_batch=next_batch)
        rewards = jnp.ones((num_envs,), dtype=jnp.float32)
        step_output = DummyStepOutput(
            prev_states=qpos,
            next_states=qpos + 1.0,
            rewards=rewards,
            terminals=terminals,
            commanded_ctrl=actions,
            applied_ctrl=actions,
            actual_wrench=jnp.zeros((num_envs, 1), dtype=jnp.float32),
            desired_wrench=jnp.zeros((num_envs, 1), dtype=jnp.float32),
            prev_obs=qpos,
            next_obs=qpos + 1.0,
            success_terminals=terminals,
            failure_terminals=jnp.zeros_like(terminals),
            reward_components={},
        )
        return next_state, step_output, step_output.next_states

    def _vecenv_reset_stub(state, _config, mask):
        reset_qpos = jnp.zeros_like(state.mjx_batch.qpos)
        return state.replace(
            mjx_batch=state.mjx_batch.replace(
                qpos=jnp.where(mask[:, None], reset_qpos, state.mjx_batch.qpos)
            )
        )

    def _prepare(_step, states, context, policy_state):
        del context
        return states, policy_state

    def _sample(_step, policy_input, rng_key, policy_state):
        actions = jnp.zeros((num_envs, 1), dtype=jnp.float32)
        values = jnp.ones((num_envs,), dtype=jnp.float32)
        logp = jnp.zeros((num_envs,), dtype=jnp.float32)
        return actions, values, logp, rng_key, policy_state

    def _post(_step, step_output, actions, context, reset_flag, policy_state):
        del step_output, actions, reset_flag
        return context, None, policy_state

    def _bootstrap(_step, env_state, context, rng_key, policy_state):
        del env_state, context, policy_state
        return jnp.full((num_envs,), 7.0, dtype=jnp.float32), rng_key, None

    result = ru.run_functional_rollout(
        step_config=step_config,
        max_episode_len=step_config.max_episode_len,
        initial_state=initial_state,
        initial_context=initial_context,
        rng=initial_state.rng,
        num_steps=num_steps,
        reference_waypoint=reference_waypoint,
        callbacks=ru.FunctionalRolloutCallbacks(
            prepare_policy_input=_prepare,
            sample_policy=_sample,
            update_context=_post,
            bootstrap_value=_bootstrap,
        ),
        state_features_fn=lambda state, _: state.mjx_batch.qpos,
        step_fn=_vecenv_step_stub,
        reset_fn=_vecenv_reset_stub,
    )

    bootstrap_vals = result.bootstrap_values
    assert jnp.all(result.terminated_masks == 0)
    assert jnp.all(result.truncated_masks[1])
    assert jnp.all(result.truncated_masks[3])
    assert not bool(result.truncated_masks[0].any())
    assert not bool(result.truncated_masks[2].any())
    assert not bool(result.truncated_masks[4].any())
    assert jnp.allclose(bootstrap_vals[1], 7.0)
    assert jnp.allclose(bootstrap_vals[3], 7.0)
    assert jnp.allclose(bootstrap_vals[4], 7.0)
    assert jnp.allclose(bootstrap_vals[0], 0.0)
    assert jnp.allclose(bootstrap_vals[2], 0.0)


def test_run_functional_rollout_records_policy_input_context() -> None:
    num_envs = 1
    num_steps = 3

    initial_batch = DummyBatch(qpos=jnp.zeros((num_envs, 1), dtype=jnp.float32))
    initial_state = DummyState(
        rng=jax.random.PRNGKey(0),
        mjx_batch=initial_batch,
    )
    step_config = DummyStepConfig(max_episode_len=10)
    initial_context = jnp.zeros((num_envs, 1), dtype=jnp.float32)
    reference_waypoint = jnp.zeros((3,), dtype=jnp.float32)

    def _vecenv_step_stub(state, actions, _waypoint, _config, _context, _features):
        qpos = state.mjx_batch.qpos
        terminals = jnp.zeros((num_envs,), dtype=bool)
        next_state = state.replace(mjx_batch=state.mjx_batch.replace(qpos=qpos + 1.0))
        step_output = DummyStepOutput(
            prev_states=qpos,
            next_states=qpos + 1.0,
            rewards=jnp.ones((num_envs,), dtype=jnp.float32),
            terminals=terminals,
            commanded_ctrl=actions,
            applied_ctrl=actions,
            actual_wrench=jnp.zeros((num_envs, 1), dtype=jnp.float32),
            desired_wrench=jnp.zeros((num_envs, 1), dtype=jnp.float32),
            prev_obs=qpos,
            next_obs=qpos + 1.0,
            success_terminals=terminals,
            failure_terminals=jnp.zeros_like(terminals),
            reward_components={},
        )
        return next_state, step_output, step_output.next_states

    def _sample(_step, policy_input, rng_key, policy_state):
        actions = policy_input
        values = jnp.zeros((num_envs,), dtype=jnp.float32)
        logp = jnp.zeros((num_envs,), dtype=jnp.float32)
        return actions, values, logp, rng_key, policy_state

    def _post(_step, step_output, actions, context, reset_flag, policy_state):
        del step_output, actions, reset_flag
        return context + 1.0, None, policy_state

    result = ru.run_functional_rollout(
        step_config=step_config,
        max_episode_len=step_config.max_episode_len,
        initial_state=initial_state,
        initial_context=initial_context,
        rng=initial_state.rng,
        num_steps=num_steps,
        reference_waypoint=reference_waypoint,
        callbacks=ru.FunctionalRolloutCallbacks(
            prepare_policy_input=lambda _step, _states, context, extra: (
                context,
                extra,
            ),
            sample_policy=_sample,
            update_context=_post,
            bootstrap_value=lambda step, state, context, key, extra: (
                jnp.zeros(num_envs),
                key,
                extra,
            ),
        ),
        state_features_fn=lambda state, _: state.mjx_batch.qpos,
        step_fn=_vecenv_step_stub,
        reset_fn=lambda state, _config, mask: state,
    )

    expected = jnp.array([[[0.0]], [[1.0]], [[2.0]]], dtype=jnp.float32)
    assert jnp.allclose(result.actions, expected)
    assert jnp.allclose(result.context, expected)
    assert jnp.allclose(result.final_context, jnp.array([[3.0]], dtype=jnp.float32))
