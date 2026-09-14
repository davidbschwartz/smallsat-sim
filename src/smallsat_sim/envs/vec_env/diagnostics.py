"""Optional runtime/backend comparison, restoring all runtime state afterwards."""
import jax
import jax.numpy as jnp
import numpy as np
from .mjx_backend import step_with_observations
from .observations import mjx_state_features


def verify_functional_step(env, state, commanded_ctrl, next_waypoint, *,
                           step_config, atol=1e-6, rtol=1e-5):
    """Compare matching transition fields and complete state trees."""
    if not step_config.effects_enabled:
        raise ValueError("Runtime parity requires effects_enabled=True")
    original = env.state_struct
    original_components = env._last_reward_components
    original_obs = getattr(env, 'obs', None)
    actions = jnp.asarray(commanded_ctrl)
    reference = jnp.asarray(next_waypoint)

    def assert_equal(name, expected, actual):
        if jax.tree.structure(expected) != jax.tree.structure(actual):
            raise AssertionError(f"{name} structure differs")
        for left, right in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
            left, right = np.asarray(left), np.asarray(right)
            if np.issubdtype(left.dtype, np.inexact):
                np.testing.assert_allclose(left, right, atol=atol, rtol=rtol, err_msg=name)
            else:
                np.testing.assert_array_equal(left, right, err_msg=name)

    try:
        expected_state, expected = step_with_observations(state, actions, reference, step_config)
        env.apply_state_struct(state)
        previous_features = mjx_state_features(state.mjx_batch, reference)
        rewards, terminals = env.transition(actions, previous_features, reference)
        assert_equal('state', expected_state, env.state_struct)
        assert_equal('previous features', expected.prev_states, previous_features)
        assert_equal('next features', expected.next_states, env.get_states(reference))
        assert_equal('rewards', expected.rewards, rewards)
        assert_equal('terminals', expected.terminals, terminals)
        assert_equal('applied control', expected.applied_ctrl, env.mjx_batch.ctrl)
        assert_equal('actual wrench', expected.actual_wrench, env.get_actual_wrench())
        assert_equal('desired wrench', expected.desired_wrench, env.get_desired_wrench(actions))
        assert_equal('components', expected.reward_components, env.get_last_reward_components())
    finally:
        env.apply_state_struct(original)
        env._last_reward_components = original_components
        if original_obs is not None:
            env.obs = original_obs
        elif hasattr(env, 'obs'):
            del env.obs
