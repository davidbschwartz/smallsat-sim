"""Custom functions execute in both compiled backends and survive realized replay."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from smallsat_sim.envs.vec_env import freeflyer, mjx_backend
from smallsat_sim.api.experiments import make_experiment
from smallsat_sim.envs.effects.custom import apply_custom_effect, CustomEffectState
from smallsat_sim.model.vehicle import load_vehicle


def make_custom(backend):
    actuator = load_vehicle('vehicles/astrobee.yaml').actuators[0].name
    return make_experiment(vehicle='astrobee_rl', failures='off', log=False, planner_radius=0,
        overrides={'RL': {'num_envs': 2, 'use_adaptive_approach': False,
            'rollout_backend': backend, 'PPO': {'steps_per_epoch': 2, 'num_minibatches': 1},
            'custom_faults': [{'sample': 'examples.custom_effects:sample_motor_loss',
                'apply': 'examples.custom_effects:apply_motor_loss', 'start_time': 0.01,
                'params': {'actuator': actuator, 'minimum': .5, 'maximum': .5}}],
            'custom_disturbances': [{'sample': 'examples.custom_effects:sample_wrench',
                'apply': 'examples.custom_effects:apply_wrench',
                'params': {'wrench': [1., 0., 0., .1, .2, .3]}}]}})


@pytest.mark.parametrize('backend', ['freeflyer', 'mjx'])
def test_custom_effect_compiled_application_and_replay(backend, tmp_path):
    experiment = make_custom(backend)
    try:
        env = experiment.env
        controls = jnp.ones((2, 12)) * .2
        original = env.state_struct
        effect = original.perturbation_states[-1]
        np.testing.assert_array_equal(apply_custom_effect(effect, controls, 0.)[0], controls)
        state = env.freeflyer_state_struct() if backend == 'freeflyer' else original
        if backend == 'freeflyer':
            state = state.replace(time=jnp.ones(2))
            prepare = jax.jit(freeflyer.prepare_effects)
        else:
            state = state.replace(mjx_batch=state.mjx_batch.replace(time=jnp.ones(2)))
            prepare = jax.jit(mjx_backend.prepare_effects)
        updated, applied, force = prepare(state, controls)
        np.testing.assert_allclose(applied[:, 0], .1)
        np.testing.assert_allclose(force[:, 0], 1.)
        torque = np.broadcast_to([.1, .2, .3], (2, 3))
        if backend == 'mjx':
            torque = np.einsum('bji,bj->bi', np.asarray(state.mjx_batch.xmat[:, 1]), torque)
        np.testing.assert_allclose(force[:, 3:], torque, atol=1e-7)
        assert np.all(np.asarray(updated.perturbation_states[-1].data['elapsed']) > 0)
        config = env.build_step_config()
        if backend == 'freeflyer':
            reset = freeflyer.reset_masked(updated, config, jnp.array([True, False]))
        else:
            reset = mjx_backend.reset(updated, config, jnp.array([True, False]))
        np.testing.assert_array_equal(reset.perturbation_states[-1].data['elapsed'],
                                      updated.perturbation_states[-1].data['elapsed'])
        original = original.replace(perturbation_states=updated.perturbation_states,
                                    disturbance_states=updated.disturbance_states)
        from smallsat_sim.controllers.rl.runners.runner_utils import save_training_data, load_training_data
        save_training_data(tmp_path, 'scenario', {'state': original})
        env.reset_perturbations()
        env.reset_disturbances()
        restored = load_training_data(tmp_path, 'scenario', mjx_batch_template=env.mjx_batch)['state']
        mismatched = restored.replace(perturbation_states=(*restored.perturbation_states[:-1],
            restored.perturbation_states[-1].replace(version='changed')))
        previous = env.state_struct
        with pytest.raises(ValueError, match='same apply function, version'):
            env.apply_state_struct(mismatched)
        for before, after in zip(jax.tree.leaves(previous), jax.tree.leaves(env.state_struct), strict=True):
            np.testing.assert_array_equal(before, after)
        env.apply_state_struct(restored)
        np.testing.assert_array_equal(restored.perturbation_states[-1].data['elapsed'],
                                      updated.perturbation_states[-1].data['elapsed'])
        np.testing.assert_array_equal(env.perturbation_states[-1].data['efficiency'], effect.data['efficiency'])
        result = experiment.runner.collect(experiment.runner.collector('zero', stochastic=False), randomize=False)
        assert isinstance(result.final_state.perturbation_states[-1], CustomEffectState)
        np.testing.assert_allclose(result.step_outputs.applied_ctrl[1, :, 0],
                                   result.actions[1, :, 0] * .5)
        env._pre_physics_step(controls)
        assert env.mjx_batch.qfrc_applied.shape == (2, 6)
    finally:
        experiment.env.close()


def bad_apply(state, values, time, dt):
    return values[:, :1], state


def test_custom_effect_contract_rejects_bad_shape_and_freezes_inactive_rows():
    from smallsat_sim.envs.effects.custom import CustomEffect, EffectSampleContext
    context = EffectSampleContext(load_vehicle('vehicles/astrobee.yaml'), 2, .05)
    spec = {'sample': 'examples.custom_effects:sample_motor_loss',
            'apply': 'examples.custom_effects:apply_motor_loss', 'probability': 0.,
            'params': {'actuator': 'thruster1', 'minimum': .5, 'maximum': .5}}
    effect = CustomEffect(spec, context, jax.random.PRNGKey(0))
    controls = jnp.ones((2, 12))
    output, advanced = jax.jit(apply_custom_effect)(effect.state, controls, jnp.ones(2))
    np.testing.assert_array_equal(output, controls)
    np.testing.assert_array_equal(advanced.data['elapsed'], [0., 0.])
    with pytest.raises(ValueError, match='preserve input shape'):
        CustomEffect({**spec, 'apply': f'{__name__}:bad_apply'}, context, jax.random.PRNGKey(0))
    # Declared order is ordinary function composition, including overlapping actuators.
    active = effect.state.replace(active_mask=jnp.ones(2, dtype=bool))
    first, _ = apply_custom_effect(active, controls, jnp.ones(2))
    second, _ = apply_custom_effect(active, first, jnp.ones(2))
    np.testing.assert_allclose(second[:, 0], .25)


@pytest.mark.parametrize('backend', ['mjx', 'freeflyer'])
@pytest.mark.parametrize('effects_enabled', [False, True])
def test_full_and_training_steps_agree_with_custom_effects(effects_enabled, backend):
    """Output adapters must preserve physics, terminal counts and reward semantics."""
    experiment = make_custom(backend)
    try:
        env = experiment.env
        operations = env.rollout_backend()
        state = env.freeflyer_state_struct() if backend == 'freeflyer' else env.state_struct
        if backend == 'freeflyer':
            state = state.replace(time=jnp.ones(2))
        else:
            state = state.replace(mjx_batch=state.mjx_batch.replace(time=jnp.ones(2)))
        config = env.build_step_config(effects_enabled=effects_enabled)
        controls = jnp.full((2, 12), .2)
        reference = jnp.array([0., 0., 0., 1., 0., 0., 0.])
        full_state, full = jax.jit(lambda state: operations.step_with_observations(
            state, controls, reference, config,
        ))(state)
        train_state, train, next_states = jax.jit(lambda state: operations.step(
            state, controls, reference, config,
        ))(state)
        for lhs, rhs in zip(jax.tree.leaves(full_state), jax.tree.leaves(train_state), strict=True):
            np.testing.assert_allclose(lhs, rhs, atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(full.next_states, next_states, atol=1e-6)
        for name in ('rewards', 'terminals', 'applied_ctrl', 'actual_wrench',
                     'success_terminals', 'failure_terminals'):
            np.testing.assert_allclose(getattr(full, name), getattr(train, name), atol=1e-6)
    finally:
        experiment.env.close()


def test_runtime_transition_wrench_reward_snapshot_and_diagnostic():
    from smallsat_sim.api.registry import register_reward
    from smallsat_sim.api.rewards import RewardResult

    def wrench_reward(context):
        values = jnp.sum(context.actual_wrench - context.desired_wrench, axis=1)
        return RewardResult(values, {'wrench': values})

    register_reward('test/vecenv_wrench_contract', wrench_reward)
    experiment = make_custom('mjx')
    try:
        env = experiment.env
        env.env_cfg.environment.reward = 'test/vecenv_wrench_contract'
        env.env_cfg.environment.control_decimation = 3
        state = env.state_struct
        state = state.replace(mjx_batch=state.mjx_batch.replace(time=jnp.array([0., 2.])))
        env.apply_state_struct(state)
        actions = jnp.full((2, 12), .2)
        reference = jnp.array([0., 0, 0, 1, 0, 0, 0])
        expected_state, expected = jax.jit(lambda state: mjx_backend.step_with_observations(
            state, actions, reference, env.build_step_config()))(state)
        previous = env.get_states(reference)
        actual, _ = env.transition(actions, previous, reference)
        np.testing.assert_allclose(actual, expected.rewards, atol=1e-6)
        np.testing.assert_allclose(env.state_struct.mjx_batch.qpos,
                                  expected_state.mjx_batch.qpos, atol=1e-6)
        saved = env.state_struct
        components = env.get_last_reward_components()
        env.verify_functional_step(state, actions, reference, step_config=env.build_step_config())
        assert env.get_last_reward_components() is components
        for left, right in zip(jax.tree.leaves(saved), jax.tree.leaves(env.state_struct), strict=True):
            np.testing.assert_array_equal(left, right)
        invalid = saved.replace(perturbation_states=saved.perturbation_states[:-1])
        with pytest.raises(ValueError):
            env.apply_state_struct(invalid)
        for left, right in zip(jax.tree.leaves(saved), jax.tree.leaves(env.state_struct), strict=True):
            np.testing.assert_array_equal(left, right)
    finally:
        experiment.env.close()


def test_viewer_substeps_match_headless_physics(monkeypatch):
    from smallsat_sim.envs.vec_env import runtime
    experiment = make_custom('mjx')
    try:
        env = experiment.env
        env.env_cfg.environment.control_decimation = 3
        env.env_cfg.viewer.viewer_decimation = 2
        state = env.state_struct
        actions = jnp.full((2, 12), .1)
        env.step(actions)
        expected = env.state_struct
        env.apply_state_struct(state)
        frames = []
        monkeypatch.setattr(runtime.mjx, 'get_data_into',
            lambda _data, _model, batch: frames.append(float(batch.time[0])))
        env.viewer = object()
        env._update_viewer = lambda: None
        env.step(actions)
        np.testing.assert_allclose(frames, [0, 2 * env.model.opt.timestep], atol=1e-7)
        for left, right in zip(jax.tree.leaves(expected), jax.tree.leaves(env.state_struct), strict=True):
            np.testing.assert_allclose(left, right, atol=1e-6, rtol=1e-5)
    finally:
        experiment.env.viewer = None
        experiment.env.close()
