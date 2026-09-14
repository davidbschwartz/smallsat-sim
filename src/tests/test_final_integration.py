"""External task/asset/reward/effects work together through rollout and resume."""
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from smallsat_sim.envs.vec_env import freeflyer, mjx_backend

from smallsat_sim.api import register_reward, RewardResult, load_vehicle
from smallsat_sim.api.experiments import make_experiment


@pytest.mark.parametrize('backend', ['freeflyer', 'mjx'])
def test_external_experiment_rollout_update_and_resume(tmp_path, backend):
    import examples.position_task
    from smallsat_sim.controllers.rl.storage.rollout_batch import make_training_batch
    from smallsat_sim.controllers.rl.runners.runner_utils import save_training_data, load_training_data

    experiment = make_experiment('examples/integrated_experiment.yaml', overrides={
        'RL': {'rollout_backend': backend, 'checkpoint_dir': str(tmp_path / 'checkpoints')},
    })
    try:
        env, runner = experiment.env, experiment.runner
        assert env.model_cfg.name == 'demo_spacecraft'
        assert env.model.body_mass[1] == 4.
        assert experiment.run_spec.task == 'position_only'
        collector = runner.collector('zero', stochastic=True)
        result = runner.collect(collector, randomize=False)
        assert bool(jnp.any(result.final_state.perturbation_states[-1].data['elapsed'] > 0))
        runner.agent.update(make_training_batch(result, runner.context_scale, runner.agent.gamma, runner.agent.lam))
        runner.save('policy')
        expected = runner.collect(collector, randomize=False)
        runner.restore('policy')
        resumed = runner.collect(collector, randomize=False)
        jax.tree.map(np.testing.assert_array_equal, expected, resumed)

        config = env.build_step_config()
        state = freeflyer.to_mjx(resumed.final_state, env.state_struct, config) if backend == 'freeflyer' else resumed.final_state
        save_training_data(tmp_path, 'realized', {'state': state})
        restored = load_training_data(tmp_path, 'realized', mjx_batch_template=env.mjx_batch)['state']
        controls = jnp.ones((env.num_envs, env.act_dim)) * .1
        expected = mjx_backend.prepare_effects(state, controls)
        actual = mjx_backend.prepare_effects(restored, controls)
        jax.tree.map(np.testing.assert_array_equal, expected, actual)
    finally:
        experiment.env.close()


def test_reward_override_is_selected_and_invalid_names_fail_before_allocation(monkeypatch):
    import smallsat_sim.api.experiments as factory
    from smallsat_sim.api.registry import RegistryError
    register_reward('integration/config_reward', lambda c: RewardResult(
        jnp.zeros(c.actions.shape[0]), {}), replace=True)
    def inspect(spec, entry, config):
        assert spec.reward == 'integration/config_reward'
        assert config.env.environment.reward == spec.reward
        raise RuntimeError('resolved correctly')
    monkeypatch.setattr(factory, '_build_env', inspect)
    with pytest.raises(RuntimeError, match='resolved correctly'):
        make_experiment(vehicle='astrobee_rl', overrides={'RL.reward': 'integration/config_reward'})
    with pytest.raises(RuntimeError, match='resolved correctly'):
        make_experiment(vehicle='astrobee_rl', reward='integration/config_reward',
                        overrides={'RL.reward': 'missing_reward'})
    with pytest.raises(RegistryError, match='Unknown reward'):
        make_experiment(vehicle='astrobee_rl', overrides={'RL.reward': 'missing_reward'})


def test_vectorized_task_rejects_multiple_bodies_before_allocation(monkeypatch):
    from smallsat_sim.envs.vehicles.astrobee_rl.env import AstrobeeEnvVectorized
    from smallsat_sim.envs.vehicles.astrobee_rl.config import resolve_config
    from smallsat_sim.envs.vec_env import VecEnv
    vehicle = load_vehicle('vehicles/cubesat.yaml')
    vehicle = replace(vehicle, bodies=(*vehicle.bodies, replace(vehicle.bodies[0], name='second')))
    def allocate(*args, **kwargs):
        raise AssertionError('Must reject before allocation')
    monkeypatch.setattr(VecEnv, '__init__', allocate)
    with pytest.raises(ValueError, match='exactly one free body'):
        AstrobeeEnvVectorized(None, config=resolve_config().env, vehicle=vehicle)


def test_rl_tuple_helper_accepts_registered_tasks_and_rejects_non_rl(monkeypatch):
    import examples.position_task
    import smallsat_sim.api.experiments as factory
    from types import SimpleNamespace
    captured = {}
    def construct(**options):
        captured.update(options)
        return SimpleNamespace(env='env', planner='planner', runner='runner')
    monkeypatch.setattr(factory, 'make_experiment', construct)
    assert factory.make_rl_experiment(vehicle='examples/position_only') == ('env', 'planner', 'runner')
    assert captured == {'vehicle': 'examples/position_only'}
    with pytest.raises(factory.RegistryError, match='rl/on_policy'):
        factory.make_rl_experiment(vehicle='cubesat')


def test_functional_import_does_not_load_runtime_environment():
    import subprocess
    import sys

    subprocess.run([sys.executable, "-c", """
import sys
from smallsat_sim.envs.vec_env.mjx_backend import step_with_observations, reset
for module in (
    'smallsat_sim.envs.vec_env.runtime',
    'smallsat_sim.envs.base_env',
    'smallsat_sim.envs.rendering.rollout',
):
    assert module not in sys.modules, module
"""], check=True, capture_output=True, text=True)
