"""Reward terms can change independently of episode endings and compiled stepping."""
from dataclasses import replace
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from smallsat_sim.api import RewardContext, RewardTerm, compose_reward, register_reward
from smallsat_sim.envs.rewards import full_pose_reward, fuel_cost
from smallsat_sim.envs.termination import TerminationResult


def context():
    config = SimpleNamespace(
        sigma_pos=1., sigma_vel=1., sigma_att=1., sigma_angvel=1.,
        w_pos=1., w_vel=2., w_att=3., w_angvel=4.,
        lam_fuel=.1, lam_speed_terminal=2., lam_ang_speed_terminal=3.,
        lam_fuel_terminal=.5, terminal_bonus=10., use_adaptive_approach=False,
        res_dim=0, collect_reward_components=True,
    )
    states = jnp.zeros((2, 12))
    success = jnp.array([True, False])
    failure = jnp.array([False, True])
    return RewardContext(states, states.at[:, 0].set(1.), jnp.ones((2, 3)), config,
                         TerminationResult(success | failure, success, failure, jnp.zeros(2, dtype=jnp.int32)))


def test_replace_one_default_cost_or_potential_preserves_other_terms_and_endings():
    ctx = context()
    baseline = full_pose_reward(ctx)
    changed = full_pose_reward(ctx, fuel=lambda c: jnp.zeros(c.actions.shape[0]))
    np.testing.assert_allclose(changed.rewards, baseline.rewards + fuel_cost(ctx), atol=1e-6)
    for name in baseline.components:
        if name not in {'penalty_fuel', 'penalty_total', 'reward_total'}:
            np.testing.assert_array_equal(changed.components[name], baseline.components[name])
    changed = full_pose_reward(ctx, position=lambda states, config: jnp.zeros(states.shape[0]))
    np.testing.assert_allclose(changed.rewards, baseline.rewards - baseline.components['shaping_pos'], atol=1e-6)
    np.testing.assert_array_equal(ctx.termination.terminals, [True, True])
    with pytest.raises(ValueError, match="Reward component"):
        full_pose_reward(ctx, fuel=lambda c: 0.)


def test_composer_weights_diagnostics_disabled_terms_and_shape_validation():
    ctx = context()
    def disabled(context):
        raise AssertionError('A disabled term must not execute or trace')
    terms = {'effort': RewardTerm(lambda c: jnp.sum(c.actions ** 2, axis=-1), -2),
             'success': lambda c: c.termination.success_terminals.astype(jnp.float32),
             'disabled': RewardTerm(disabled, 0)}
    reward = compose_reward(terms)
    terms.clear()
    values = jax.jit(lambda actions: reward(replace(ctx, actions=actions)).rewards)(ctx.actions)
    np.testing.assert_array_equal(values, [-5, -6])
    assert set(reward(ctx).components) == {'effort', 'success', 'reward_total'}
    ctx.config.collect_reward_components = False
    assert set(reward(ctx).components) == {'reward_total'}
    with pytest.raises(ValueError, match='must return shape'):
        compose_reward({'bad': lambda c: jnp.sum(c.actions)})(ctx)
    for terms in ({'reward_total': lambda c: c.actions[:, 0]},
                  {'bad': RewardTerm(lambda c: c.actions[:, 0], float('nan'))}):
        with pytest.raises(ValueError):
            compose_reward(terms)


@pytest.mark.parametrize('backend', ['freeflyer', 'mjx'])
def test_composed_reward_reaches_rollout_without_changing_termination(backend):
    from smallsat_sim.api.experiments import make_experiment
    register_reward('test_composition', compose_reward({
        'effort': RewardTerm(lambda c: jnp.sum(c.actions ** 2, axis=-1), -2),
        'alive': lambda c: jnp.ones(c.actions.shape[0]),
    }), replace=True)
    experiment = make_experiment(vehicle='astrobee_rl', reward='test_composition',
        failures='off', log=False, planner_radius=0,
        overrides={'RL': {'num_envs': 2, 'use_adaptive_approach': False,
            'rollout_backend': backend, 'PPO': {'steps_per_epoch': 2, 'num_minibatches': 1}}})
    try:
        result = experiment.runner.collect(experiment.runner.collector('zero', stochastic=False), randomize=False)
        np.testing.assert_allclose(result.step_outputs.rewards,
            1 - 2 * np.sum(np.asarray(result.actions) ** 2, axis=-1), atol=1e-6)
        assert experiment.env.env_cfg.environment.termination == 'full_pose'
        assert experiment.runner.resolved_config['rl']['reward'] == 'test_composition'
    finally:
        experiment.env.close()


def test_compute_penalties_residual_clip_and_terminal_terms() -> None:
    prev_states = jnp.zeros((2, 12), dtype=jnp.float32)
    next_states = prev_states.at[:, 6].set(1.0)
    terminals = jnp.array([True, False])
    commanded_ctrl = jnp.array([[1.0, -1.0], [0.5, 0.5]], dtype=jnp.float32)
    prev_context = jnp.array([[3.0, 4.0], [0.0, 0.0]], dtype=jnp.float32)

    config = SimpleNamespace(
        sigma_pos=1., sigma_vel=1., sigma_att=1., sigma_angvel=1.,
        w_pos=1., w_vel=1., w_att=1., w_angvel=1.,
        lam_fuel=1., lam_speed_terminal=2., lam_ang_speed_terminal=0.,
        lam_fuel_terminal=3., terminal_bonus=1., lam_wrench_residual=4.,
        wrench_residual_tolerance=1., wrench_residual_clip=2., res_dim=2,
        use_adaptive_approach=True, collect_reward_components=True,
    )
    details = full_pose_reward(RewardContext(
        prev_states, next_states, commanded_ctrl, config,
        TerminationResult(terminals, terminals, jnp.zeros_like(terminals),
                          jnp.zeros(2, dtype=jnp.int32)),
        prev_residuals=prev_context,
    )).components
    penalties = details["penalty_total"]

    expected = jnp.array([18.0, 1.0], dtype=jnp.float32)
    assert jnp.allclose(penalties, expected)
    assert jnp.allclose(details["penalty_fuel"], jnp.array([2.0, 1.0]))
    assert jnp.allclose(details["penalty_terminal_speed"], jnp.array([2.0, 0.0]))
    assert jnp.allclose(details["penalty_terminal_fuel"], jnp.array([6.0, 0.0]))
    assert jnp.allclose(details["penalty_wrench_residual"], jnp.array([8.0, 0.0]))
