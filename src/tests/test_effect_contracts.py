"""Behavioral contracts for state ownership and batched activation."""
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from smallsat_sim.envs.effects import preparation
from smallsat_sim.envs.effects.disturbances import ConstantForceDisturbance
from smallsat_sim.envs.effects.actuator_kernels import (
    BatchedFaultState, stuck_off_apply_from_state, stuck_on_apply_from_state,
    gp_apply_from_state,
)


def config():
    return SimpleNamespace(environment=SimpleNamespace(num_envs=2),
                           sim=SimpleNamespace(verbose=False))


def vehicle():
    return SimpleNamespace(actuators=[SimpleNamespace(forcerange=(0., 1.),
                                                       ctrlrange=(0., 1.))] * 2)


@pytest.mark.parametrize('kind', [1, 2, 3])
def test_per_environment_time_when_environment_and_actuator_counts_match(kind):
    state = BatchedFaultState(
        jax.random.PRNGKey(0), jnp.full((2, 2), kind), kind,
        jnp.ones((2, 2)), jnp.full((2, 2), .8),
        jnp.tile(jnp.array([0., 1.]), (2, 1)),
        jnp.tile(jnp.array([0., .5]), (2, 1)),
    )
    kernel = {1: stuck_off_apply_from_state, 2: stuck_on_apply_from_state,
              3: lambda s, c, t: gp_apply_from_state(s, c, t, 3)}[kind]
    expected = np.array([[.4, .4], [{1: 0., 2: .8, 3: .2}[kind]] * 2])
    for apply in (kernel, jax.jit(kernel)):
        actual, _ = apply(state, jnp.full((2, 2), .4), jnp.array([0., 2.]))
        np.testing.assert_allclose(actual, expected)


def test_restored_disturbance_drives_application_and_next_activation():
    effect = ConstantForceDisturbance(config(), jax.random.PRNGKey(0))
    effect.state = replace(effect.state, rng=jax.random.PRNGKey(42),
                           active_mask=jnp.array([True, False]),
                           start_times=jnp.array([1., 3.]),
                           params={'const_force': jnp.full((2, 6), .25)})
    np.testing.assert_array_equal(effect.apply(2.), [[.25] * 6, [0.] * 6])
    effect.const_force_disturbance(jnp.array([1]), start_time=4.)
    np.testing.assert_array_equal(effect.apply(5.), np.full((2, 6), .25))
    np.testing.assert_array_equal(effect.state.start_times, [1., 4.])
    effect.deactivate_const_force_disturbance()
    np.testing.assert_array_equal(effect.apply(5.), np.zeros((2, 6)))


def test_named_fault_lookup_is_independent_of_collection_order():
    off = preparation.StuckOffThrusters(config(), vehicle(), jax.random.PRNGKey(0))
    on = preparation.StuckOnThrusters(config(), vehicle(), jax.random.PRNGKey(1))
    collection = preparation.BatchedFaultCollection([on, off])
    collection.named('stuck_off').activate(jax.random.PRNGKey(2), jnp.array([0]), 1)
    np.testing.assert_array_equal(off.apply(jnp.ones((2, 2))), [[1., 0.], [1., 1.]])


def test_python_float_valve_bounds_reach_sampler(monkeypatch):
    captured = []
    class Sampler:
        def __init__(self, **kwargs):
            captured.append(kwargs)
        def generate_failure_data(self, key, mode):
            return jnp.array([0., 1.]), jnp.array([0., .5])
    monkeypatch.setattr(preparation, 'ThrusterFailureSimulator', Sampler)
    effect = preparation.FaultyValve(config(), vehicle(), jax.random.PRNGKey(0))
    effect.activate(jax.random.PRNGKey(1), jnp.array([0]), 0,
                    valve_min=.23, valve_max=.67)
    assert captured[0]['valve_min'] == .23
    assert captured[0]['valve_max'] == .67


def test_occupancy_tracks_restoration_without_separate_synchronization():
    effect = preparation.StuckOffThrusters(config(), vehicle(), jax.random.PRNGKey(0))
    initial = effect.state
    effect.activate(jax.random.PRNGKey(1), jnp.array([0]), 1)
    assert int(effect.occupancy.mask[0, 1]) == 1
    effect.restore_state(initial)
    np.testing.assert_array_equal(effect.occupancy.mask, np.zeros((2, 2)))


def test_checkpoint_normalizes_legacy_actuator_vectors():
    from smallsat_sim.envs.effects.actuator_kernels import (
        perturbation_state_from_serializable, perturbation_state_to_serializable,
    )
    state = BatchedFaultState(jax.random.PRNGKey(0), jnp.full((2, 2), 2), 2,
                              jnp.array([1., 3.]), jnp.array([.2, .7]))
    restored = perturbation_state_from_serializable(perturbation_state_to_serializable(state))
    np.testing.assert_array_equal(restored.start_times, [[1., 3.], [1., 3.]])
    actual, _ = stuck_on_apply_from_state(restored, jnp.ones((2, 2)), 2.)
    np.testing.assert_allclose(actual, [[.2, 1.], [.2, 1.]])


def test_kernel_rejects_ambiguous_actuator_state_shape():
    state = BatchedFaultState(jax.random.PRNGKey(0), jnp.ones((2, 2)), 1,
                              jnp.ones(2))
    with pytest.raises(ValueError, match='onset times'):
        stuck_off_apply_from_state(state, jnp.ones((2, 2)), 2.)
