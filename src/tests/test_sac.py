"""SAC numerical, replay, collection and continuation contracts."""
from types import SimpleNamespace
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import pytest

from smallsat_sim.api.experiments import make_experiment
from smallsat_sim.controllers.rl.algorithms.sac import bellman_target
from smallsat_sim.controllers.rl.modules.sac_policy import SACActor
from smallsat_sim.controllers.rl.storage.replay_buffer import Transition, empty_replay, insert, sample


def test_replay_wrap_and_valid_sampling():
    replay = empty_replay(5, 1, 1)
    def rows(start, count):
        x = jnp.arange(start, start + count, dtype=jnp.float32)
        return Transition(x[:, None], x[:, None], x, x[:, None] + 1,
                          jnp.zeros(count, bool), jnp.zeros(count, bool))
    replay = insert(replay, rows(0, 3))
    assert set(np.asarray(sample(replay, jax.random.PRNGKey(1), 100).rewards)) <= {0, 1, 2}
    replay = insert(replay, rows(3, 4))
    assert int(replay.size) == 5 and int(replay.cursor) == 2
    assert set(np.asarray(replay.data.rewards)) == {2, 3, 4, 5, 6}
    with pytest.raises(ValueError, match='exceeds'):
        insert(replay, rows(0, 6))


def test_targets_and_policy_gradients():
    target = bellman_target(jnp.array([1., 1.]), jnp.array([True, False]),
                            jnp.array([4., 4.]), jnp.array([-2., -2.]), .9, .5)
    np.testing.assert_allclose(target, [1., 5.5])
    actor = SACActor(2, [0., 2.], [1., 5.], [8], rngs=nnx.Rngs(1))
    obs = jnp.array([[1e3, -1e3], [0., 0.]])
    draw = actor.sample(obs, jax.random.PRNGKey(5))
    assert np.isfinite(draw.logp).all()
    assert np.all(draw.actions >= actor.act_low) and np.all(draw.actions <= actor.act_high)
    grads = nnx.grad(lambda a: a.sample(obs, jax.random.PRNGKey(5)).logp.mean())(actor)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(grads))


def experiment(tmp_path, backend, randomized=False):
    return make_experiment(vehicle='astrobee_rl', algorithm='sac', planner_radius=0.,
        log=False, run_name='sac_test', overrides={'RL': {
            'num_envs': 2, 'rollout_backend': backend, 'train_with_failures': randomized,
            'failure_fraction': 1.0 if randomized else 0.5,
            'disturbance_fraction': 1.0 if randomized else 0.1,
            'policy_hidden_sizes': [8, 8], 'checkpoint_dir': str(tmp_path),
            'episode_len': 4, 'n_evals': 1,
            'SAC': {'total_transitions': 8, 'collection_steps': 2,
                    'replay_capacity': 16, 'batch_size': 4, 'random_steps': 4,
                    'learning_starts': 4, 'updates_per_collection': 1, 'max_ep_len': 3,
                    'randomization_pool_size': 4}}})


def assert_tree_close(a, b):
    la, ta = jax.tree.flatten(a)
    lb, tb = jax.tree.flatten(b)
    assert ta == tb
    for x, y in zip(la, lb):
        np.testing.assert_allclose(x, y, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('backend', ['freeflyer', 'mjx'])
def test_sac_train_resume_and_actor_export(tmp_path, backend):
    exp = experiment(tmp_path, backend)
    runner = exp.runner
    try:
        runner.learn()
        assert runner.transitions == 8 and runner.gradient_updates == 2
        collector = runner.collector(stochastic=True, mode='training')
        def advance():
            carry, result = collector(runner.carry, nnx.state(runner.agent.actor),
                                      runner.reference_point, runner.transitions, 0)
            batch = sample(runner.replay, runner.replay_key, 4)
            metrics = runner.agent.update(batch)
            return carry, result, batch, metrics, nnx.state(runner.agent.objects())
        first = advance()
        runner.restore()
        second = advance()
        assert_tree_close(first, second)
        assert len(runner.evaluate()) == 1
        from smallsat_sim.controllers.rl.runners.runner_utils import load_trained_modules
        payload = load_trained_modules(tmp_path, runner.policy_file_name)
        assert set(payload) == {'actor_model', 'metadata'}
        from smallsat_sim.controllers.rl.controller import RLController
        ctrl = RLController(exp.env, exp.planner, config=runner.training_cfg,
                            checkpoint=tmp_path / runner.policy_file_name)
        ctrl._load_policy()
        assert not hasattr(ctrl.agent, 'critics')
        assert_tree_close(nnx.state(ctrl.agent.actor), nnx.state(runner.agent.actor))
        ctrl.deployment_len = 2
        ctrl.control()
        from smallsat_sim.controllers.rl.runners.runner_utils import _save
        _save(tmp_path, 'renamed.ckpt', payload)
        runner.training_state_file_name = 'renamed.ckpt'
        runner.restore_for_evaluation()
    finally:
        exp.env.close()


def test_sac_rejects_adaptation_before_allocation():
    from smallsat_sim.envs.vehicles.astrobee_rl.config import resolve_config
    with pytest.raises(ValueError, match='adaptation'):
        resolve_config(algorithm='sac', use_adaptive_approach=True)


def test_collector_keeps_terminal_next_features_and_chunk_continuity():
    from smallsat_sim.controllers.rl.runners.rollout.off_policy import (
        initial_carry, make_off_policy_collector)
    from smallsat_sim.envs.vec_env.types import VecEnvTrainingStepOutput
    class Backend:
        def state_features(self, state, reference):
            return state
        def step(self, state, actions, reference, config, context, features):
            next_state = state + 1
            terminals = (next_state[:, 0] == 2) & (jnp.arange(2) == 0)
            output = VecEnvTrainingStepOutput(
                state, next_state, jnp.zeros(2), jnp.zeros(2), jnp.zeros(2),
                jnp.ones(2), terminals, actions, jnp.zeros((2, 6)),
                terminals, jnp.zeros(2, bool), {})
            return next_state, output, next_state
        def reset(self, state, config, mask):
            return jnp.where(mask[:, None], 0., state)
    backend = Backend()
    env = SimpleNamespace(num_envs=2, act_dim=1, rollout_backend=lambda: backend,
                          build_step_config=lambda: None)
    actor = SACActor(1, [0.], [1.], [4], rngs=nnx.Rngs(0))
    carry = initial_carry(jnp.zeros((2, 1)), backend, None, jax.random.PRNGKey(1), 2)
    short = make_off_policy_collector(env, actor, steps=2, max_ep_len=3)
    long = make_off_policy_collector(env, actor, steps=4, max_ep_len=3)
    half, first = short(carry, nnx.state(actor), None, 0, 0)
    end, second = short(half, nnx.state(actor), None, 4, 0)
    whole, result = long(carry, nnx.state(actor), None, 0, 0)
    assert_tree_close(end, whole)
    assert_tree_close(jax.tree.map(lambda a, b: jnp.concatenate((a, b)), first, second), result)
    assert first.transitions.terminated[1, 0]
    assert first.transitions.next_observations[1, 0, 0] == 2
    assert second.transitions.observations[0, 0, 0] == 0
    assert second.transitions.truncated[0, 1]
    assert second.transitions.next_observations[0, 1, 0] == 3
    assert second.transitions.observations[1, 1, 0] == 0


@pytest.mark.parametrize('backend', ['freeflyer', 'mjx'])
def test_randomized_sac_fresh_runner_resume(tmp_path, backend):
    exp = experiment(tmp_path, backend, randomized=True)
    runner = exp.runner
    try:
        runner.learn()
        def advance(runner):
            bank = runner.effect_pool
            collector = runner.collector(stochastic=True, mode='training')
            return collector(runner.carry, nnx.state(runner.agent.actor),
                             runner.reference_point, runner.transitions, 0, bank)
        expected = advance(runner)
    finally:
        exp.env.close()
    resumed = experiment(tmp_path, backend, randomized=True)
    try:
        resumed.runner.restore()
        assert_tree_close(expected, advance(resumed.runner))
    finally:
        resumed.env.close()


def test_sac_target_averaging_and_temperature_direction(tmp_path):
    exp = experiment(tmp_path, 'freeflyer')
    try:
        agent = exp.runner.agent
        obs = jnp.zeros((4, exp.env.obs_dim))
        batch = Transition(obs, jnp.zeros((4, exp.env.act_dim)), jnp.ones(4), obs,
                           jnp.ones(4, bool), jnp.zeros(4, bool))
        target_before = jax.tree.map(jnp.copy, nnx.state(agent.targets))
        assert_tree_close(target_before, nnx.state(agent.critics))
        assert any(not np.array_equal(a, b) for a, b in zip(
            jax.tree.leaves(nnx.state(agent.critics.q1)), jax.tree.leaves(nnx.state(agent.critics.q2))))
        agent.target_entropy = 1000.  # Current entropy is below target: increase alpha.
        alpha_before = float(jnp.exp(agent.temperature.log_alpha.value))
        metrics = agent.update(batch)
        assert float(metrics['alpha']) > alpha_before
        expected = jax.tree.map(lambda old, new: (1-agent.tau)*old + agent.tau*new,
                                target_before, nnx.state(agent.critics))
        assert_tree_close(expected, nnx.state(agent.targets))
        assert all(np.isfinite(x) for x in metrics.values())
    finally:
        exp.env.close()


def test_per_environment_gp_reset_preserves_other_lane():
    from smallsat_sim.envs.effects.actuator_kernels import BatchedFaultState, gp_apply_from_state
    from smallsat_sim.controllers.rl.runners.rollout.off_policy_effects import merge_effects
    from dataclasses import replace
    current = BatchedFaultState(
        rng=jax.random.PRNGKey(1), thruster_mask=jnp.full((2, 1), 3), failure_value=3,
        start_times=jnp.zeros((2, 1)),
        gp_x_samples=jnp.broadcast_to(jnp.array([0., 1.]), (2, 1, 2)),
        gp_y_samples=jnp.broadcast_to(jnp.array([0., 1.]), (2, 1, 2)))
    candidate = replace(current, rng=jax.random.PRNGKey(7), gp_y_samples=current.gp_y_samples * .2)
    merged, = merge_effects((candidate,), (current,), jnp.array([True, False]))
    result, _ = gp_apply_from_state(merged, jnp.full((2, 1), .5), jnp.zeros(2))
    np.testing.assert_allclose(result[:, 0], [.1, .5])
    np.testing.assert_array_equal(merged.rng, current.rng)


@pytest.mark.parametrize('corruption', ['actions', 'cursor'])
def test_corrupt_replay_rejected_before_model_mutation(tmp_path, corruption):
    from smallsat_sim.controllers.rl.runners.runner_utils import load_trained_modules, _save
    exp = experiment(tmp_path, 'freeflyer')
    try:
        runner = exp.runner
        runner.learn()
        payload = load_trained_modules(tmp_path, runner.training_state_file_name)
        if corruption == 'actions':
            payload['replay']['data']['actions'] = jnp.zeros((2, 1))
        else:
            payload['replay']['cursor'] = jnp.array(0.5)
        _save(tmp_path, runner.training_state_file_name, payload)
        runner.agent.update(sample(runner.replay, jax.random.PRNGKey(0), 4))
        before = jax.tree.map(jnp.copy, nnx.state(runner.agent.objects()))
        with pytest.raises(ValueError, match='replay'):
            runner.restore()
        assert_tree_close(before, nnx.state(runner.agent.objects()))
    finally:
        exp.env.close()


@pytest.mark.parametrize('name', ['sac_nominal', 'sac_randomized'])
def test_packaged_sac_experiment(name):
    exp = make_experiment(f'experiments/{name}.yaml')
    try:
        assert exp.runner.agent.algorithm == 'sac'
        assert exp.runner.replay is None
        assert exp.env.num_envs == 16
        assert not exp.env.use_adaptive_approach
        assert exp.env.train_with_failures == (name == 'sac_randomized')
    finally:
        exp.env.close()


def test_sac_benchmark_evaluation_does_not_restore_replay(tmp_path, monkeypatch):
    from experiments.rl_benchmarking import evaluation
    exp = experiment(tmp_path, 'freeflyer')
    try:
        runner = exp.runner
        runner.learn()
        runner.replay = None
        monkeypatch.setattr(evaluation, 'CORE', (
            evaluation.Scenario('stuck_on', ('stuck_on',), onset_seconds=0.),))
        rows = evaluation.evaluate_suite(runner, output=tmp_path / 'scenarios')
        assert len(rows) == 1 and rows[0]['scenario'] == 'stuck_on'
        assert runner.replay is None
        assert np.isfinite(rows[0]['mean_episodic_returns'])
    finally:
        exp.env.close()


def test_ppo_checkpoint_identity_excludes_inactive_sac_defaults(tmp_path):
    from smallsat_sim.controllers.rl.runners.on_policy_runner import OnPolicyRunner
    exp = make_experiment(vehicle='astrobee_rl', algorithm='ppo', planner_radius=0.,
        log=False, overrides={'RL': {'num_envs': 2, 'rollout_backend': 'freeflyer',
            'use_adaptive_approach': False, 'policy_hidden_sizes': [8],
            'checkpoint_dir': str(tmp_path)}})
    try:
        runner = exp.runner
        assert 'SAC' not in runner.resolved_config['rl']
        runner.training_cfg.SAC.batch_size *= 2
        other = OnPolicyRunner(exp.env, exp.planner, config=runner.training_cfg)
        assert other.config_id == runner.config_id
    finally:
        exp.env.close()


def test_explicit_evaluation_checkpoint_wins_over_sibling_actor(tmp_path):
    from pathlib import Path
    from smallsat_sim.controllers.rl.runners.runner_utils import _save, model_fingerprint
    exp = experiment(tmp_path, 'freeflyer')
    try:
        runner = exp.runner
        metadata = {'config_id': runner.config_id}
        saved = nnx.state(runner.agent.actor)
        expected = model_fingerprint(runner.agent.actor)
        # A stale companion has a valid config but different weights.
        stale = nnx.clone(runner.agent.actor)
        params = nnx.state(stale, nnx.Param)
        nnx.update(stale, jax.tree.map(lambda x: x + .5, params))
        _save(tmp_path, runner.training_state_file_name, {'actor_model': saved, 'metadata': metadata})
        _save(tmp_path, runner.policy_file_name, {'actor_model': nnx.state(stale), 'metadata': metadata})
        runner.evaluation_checkpoint = Path(tmp_path) / runner.training_state_file_name
        runner.restore_for_evaluation()
        assert model_fingerprint(runner.agent.actor) == expected
    finally:
        exp.env.close()


def test_randomized_pool_preserves_configured_fault_environment_ids(tmp_path):
    from dataclasses import asdict
    from smallsat_sim.envs.effects.catalog import FaultSpec
    from smallsat_sim.model.vehicle import load_vehicle
    actuator = load_vehicle('vehicles/astrobee.yaml').actuators[0].name
    exp = make_experiment(vehicle='astrobee_rl', algorithm='sac', log=False, planner_radius=0.,
        overrides={'RL': {
            'num_envs': 2, 'rollout_backend': 'freeflyer', 'train_with_failures': True,
            'failure_fraction': 0., 'disturbance_fraction': 0.,
            'faults': [asdict(FaultSpec('stuck_off', actuator=actuator, env_ids=(0,)))],
            'policy_hidden_sizes': [4], 'checkpoint_dir': str(tmp_path),
            'SAC': {'collection_steps': 8, 'max_ep_len': 1, 'randomization_pool_size': 4},
        }})
    try:
        runner = exp.runner
        carry = runner._initial_carry(True)
        pool = runner._make_effect_pool()
        collector = runner.collector(stochastic=False)
        _, result = collector(carry, nnx.state(runner.agent.actor), runner.reference_point, 0, 0, pool)
        np.testing.assert_array_equal(result.step_outputs.applied_ctrl[:, 0, 0], 0.)
        assert np.all(np.asarray(result.step_outputs.applied_ctrl[:, 1, 0]) > 0.)
    finally:
        exp.env.close()
