"""Termination owns success/failure masks and hold-counter transitions."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from smallsat_sim.envs.termination import TerminationContext, full_pose_termination


def config():
    return SimpleNamespace(
        terminal_radius=.25, terminal_max_att_error=.25, terminal_hold_steps=3,
        enable_failure_termination=True, failure_max_position_error=8,
        failure_max_speed=2, failure_max_att_error=2.8, failure_max_ang_speed=2,
    )


def test_hold_counter_updates_once_and_resets_outside_goal():
    states = jnp.zeros((3, 12)).at[1, 0].set(1).at[2, 0].set(9)
    counts = jnp.array([2, 2, 1], dtype=jnp.int32)
    @jax.jit
    def evaluate(states, counts):
        result = full_pose_termination(TerminationContext(states, counts, config()))
        return result.terminals, result.success_terminals, result.failure_terminals, result.terminal_hold_counts
    terminated, success, failure, counts = evaluate(states, counts)
    np.testing.assert_array_equal(success, [True, False, False])
    np.testing.assert_array_equal(failure, [False, False, True])
    np.testing.assert_array_equal(terminated, [True, False, True])
    np.testing.assert_array_equal(counts, [3, 0, 0])


def test_success_requires_consecutive_pose_matches_but_not_low_velocity():
    cfg = config()
    cfg.enable_failure_termination = False
    # Position/attitude at the boundary qualify; either outside resets the hold.
    # Speed is a failure criterion when enabled, not a success criterion.
    states = jnp.zeros((4, 12)).at[0, 0].set(.25).at[0, 3].set(.25)
    states = states.at[1, 0].set(.251).at[2, 3].set(.251)
    states = states.at[3, 6].set(100.).at[3, 9].set(100.)
    counts = jnp.zeros(4, dtype=jnp.int32)
    for step in range(1, 4):
        result = full_pose_termination(TerminationContext(states, counts, cfg))
        np.testing.assert_array_equal(result.success_terminals,
                                      [step == 3, False, False, step == 3])
        np.testing.assert_array_equal(result.terminal_hold_counts, [step, 0, 0, step])
        np.testing.assert_array_equal(result.failure_terminals, [False] * 4)
        counts = result.terminal_hold_counts
