"""Effect composition and native callbacks preserve ordering and backend identities."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from smallsat_sim.envs.effects import classical, preparation
from smallsat_sim.envs.effects.disturbances import DisturbanceList, DisturbanceStatus
from smallsat_sim.envs.effects.catalog import gp_failure_modes
from smallsat_sim.envs.effects.actuator_kernels import BatchedFaultState


@pytest.mark.parametrize('module, collection', [
    (classical, classical.PerturbationList),
    (preparation, preparation.BatchedFaultCollection),
])
def test_actuator_composition_and_callback_skip_custom_effects(module, collection):
    calls = []
    effects = collection([
        SimpleNamespace(apply=lambda values, time: values * 2),
        SimpleNamespace(failure_type=module.PerturbationStatus.STUCK_OFF,
                        apply=lambda values, time: values + time,
                        key_callback=calls.append),
    ])
    np.testing.assert_array_equal(effects.apply(jnp.array([1., 2.]), 3.), [5., 7.])
    effects.key_callback(np.int64(ord(' ')))
    effects.key_callback(None)
    effects.key_callback(ord('?'))
    assert calls == [ord(' ')]
    assert ';' not in effects.keycode_dict


def test_disturbance_composition_adds_wrenches_and_dispatches():
    calls = []
    effects = DisturbanceList([
        SimpleNamespace(apply=lambda timestamp: jnp.ones(6)),
        SimpleNamespace(failure_type=DisturbanceStatus.CONSTANT_FORCE,
                        apply=lambda timestamp: jnp.full(6, timestamp),
                        key_callback=calls.append),
    ])
    np.testing.assert_array_equal(effects.apply(2.), np.full(6, 3.))
    effects.key_callback(ord('/'))
    assert calls == [ord('/')]


def test_gp_defaults_are_fresh_and_status_ids_are_preserved():
    classic = gp_failure_modes(classical.PerturbationStatus)
    batched = gp_failure_modes(preparation.PerturbationStatus, with_jitter=True)
    for name in ('FAULTY_VALVE', 'SATURATED_THRUST', 'THRUST_INSTABILITY'):
        original = classic[classical.PerturbationStatus[name]]
        assert {k: v for k, v in batched[preparation.PerturbationStatus[name]].items()
                if k != 'jitter'} == original
    assert classical.PerturbationStatus.FAULTY_VALVE.value == 4
    assert preparation.PerturbationStatus.FAULTY_VALVE.value == 3
    classic[classical.PerturbationStatus.FAULTY_VALVE]['lengthscale'] = -1
    assert gp_failure_modes(classical.PerturbationStatus)[classical.PerturbationStatus.FAULTY_VALVE]['lengthscale'] == .3


def test_state_pytree_keeps_dispatch_static_and_serialization_out_of_tracing():
    state = BatchedFaultState(jax.random.PRNGKey(0), jnp.ones((2, 3)), failure_value=3)
    leaves, structure = jax.tree.flatten(state)
    restored = jax.tree.unflatten(structure, leaves)
    assert restored.failure_value == 3 and restored.start_times is None
    changed = jax.jit(lambda value: jax.tree.map(lambda array: array + 1, value))(state)
    assert changed.failure_value == 3
    np.testing.assert_array_equal(changed.thruster_mask, np.full((2, 3), 2))
