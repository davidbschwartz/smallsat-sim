"""Named mixtures preserve sampler ordering and reject invalid inputs early."""
import numpy as np
import pytest
from smallsat_sim.envs.effects.catalog import FAULT_NAMES, fault_weights
from smallsat_sim.envs.vehicles.astrobee_rl.config import resolve_config


def test_named_weights_use_fixed_effect_order_and_omit_unselected_effects():
    np.testing.assert_array_equal(
        fault_weights({'thrust_instability': 3, 'stuck_off': 2}), [2, 0, 0, 0, 3],
    )
    np.testing.assert_array_equal(
        fault_weights(dict(reversed(list(zip(FAULT_NAMES, range(1, 6)))))),
        [1, 2, 3, 4, 5],
    )


@pytest.mark.parametrize('value', [[], [1, 0, 0, 0, 0], {}, {'stuck_off': -1},
    {'stuck_off': float('nan')}, {'stuck_on': float('inf')}, {'stuck_off': [1]},
    {'typo': 1}, {'stuck_off': 0}])
def test_invalid_mixtures_are_rejected(value):
    with pytest.raises(ValueError):
        fault_weights(value)


def test_config_overrides_named_weights_and_validates_before_construction():
    bundle = resolve_config(overrides={'RL': {'failure_distribution': {'stuck_off': 2}}})
    assert bundle.env.environment.failure_distribution.stuck_off == 2
    assert bundle.env.environment.failure_distribution.stuck_on == .2
    restored = resolve_config(overrides={'RL': {'failure_distribution': [1, 0, 0, 0, 0]}})
    assert restored.env.environment.failure_distribution == dict(zip(FAULT_NAMES, [1, 0, 0, 0, 0]))
    with pytest.raises(ValueError, match='exactly five'):
        resolve_config(overrides={'RL': {'failure_distribution': [1, 0]}})
    with pytest.raises(ValueError, match='Unknown override'):
        resolve_config(overrides={'RL': {'failure_distribution': {'typo': 1}}})
    with pytest.raises(ValueError, match='positive sum'):
        resolve_config(overrides={'RL': {'failure_distribution': dict.fromkeys(FAULT_NAMES, 0)}})


def test_named_mixture_drives_real_fault_sampling_and_checkpoint_identity():
    import jax
    from smallsat_sim.api.experiments import make_experiment
    from smallsat_sim.envs.effects.actuator_kernels import PerturbationStatus

    distribution = dict.fromkeys(FAULT_NAMES, 0.)
    distribution['stuck_off'] = 1.
    experiment = make_experiment(
        vehicle='astrobee_rl', failures='off', log=False, planner_radius=0,
        overrides={'RL': {'num_envs': 3, 'use_adaptive_approach': False,
            'rollout_backend': 'freeflyer', 'failure_distribution': distribution,
            'PPO': {'steps_per_epoch': 2, 'num_minibatches': 1}}},
    )
    try:
        env = experiment.env
        env.schedule_random_faults(jax.random.PRNGKey(5), 1., start_time=2)
        mask = np.asarray(env.thruster_occupancy.mask)
        np.testing.assert_array_equal((mask == PerturbationStatus.STUCK_OFF.value).sum(axis=1), [1, 1, 1])
        assert set(np.unique(mask)) == {PerturbationStatus.OPERATIONAL.value, PerturbationStatus.STUCK_OFF.value}
        assert experiment.runner.resolved_config['rl']['failure_distribution'] == [1., 0., 0., 0., 0.]
        before = mask.copy()
        with pytest.raises(ValueError, match='positional'):
            env.schedule_random_faults(jax.random.PRNGKey(6), 1., [0, 1, 0, 0, 0])
        np.testing.assert_array_equal(env.thruster_occupancy.mask, before)
        env.reset_perturbations()
        env.schedule_random_faults(jax.random.PRNGKey(7), 1., {'stuck_on': 1})
        mask = np.asarray(env.thruster_occupancy.mask)
        np.testing.assert_array_equal((mask == PerturbationStatus.STUCK_ON.value).sum(axis=1), [1, 1, 1])
    finally:
        experiment.env.close()
