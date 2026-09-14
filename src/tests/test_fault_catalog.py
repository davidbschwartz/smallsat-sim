"""Built-in fault catalog registration and application tests."""

from types import ModuleType, SimpleNamespace
import sys

import jax
import jax.numpy as jnp
import numpy as np

from smallsat_sim.envs.effects.catalog import (
    BUILTIN_FAULTS, FAULTS_BY_STATUS, BuiltinFaultDefinition,
)
from smallsat_sim.envs.effects.actuator_kernels import BatchedFaultState
from smallsat_sim.envs.effects.catalog import apply_actuator_effects
from smallsat_sim.envs.effects.preparation import create_actuator_effects


def test_catalog_constructs_all_modes_in_checkpoint_order():
    cfg = SimpleNamespace(environment=SimpleNamespace(num_envs=2),
                          sim=SimpleNamespace(verbose=False))
    vehicle = SimpleNamespace(actuators=[SimpleNamespace(forcerange=(0., 1.),
                                                         ctrlrange=(0., 1.))])
    keys = jax.random.split(jax.random.PRNGKey(0), len(BUILTIN_FAULTS))
    occupancy, effects = create_actuator_effects(cfg, vehicle, keys)
    assert len({item.name for item in BUILTIN_FAULTS}) == len(BUILTIN_FAULTS)
    assert [item.status_id for item in BUILTIN_FAULTS] == [1, 2, 3, 4, 5]
    for definition, effect in zip(BUILTIN_FAULTS, effects.perturbations, strict=True):
        assert effects.named(definition.name) is effect
        assert effect.state.failure_value == definition.status_id
        assert effect.occupancy is occupancy
        np.testing.assert_array_equal(effect.apply(jnp.ones((2, 1))), [[1.], [1.]])


def test_new_kernel_dispatches_without_changing_rollout_logic(monkeypatch):
    module = ModuleType('test_registered_fault')
    module.apply = lambda state, control, time: (control * .25, state)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    definition = BuiltinFaultDefinition(SimpleNamespace(value=91, name='QUARTER_THRUST'),
                                        'UnusedAdapter', module.apply)
    monkeypatch.setitem(FAULTS_BY_STATUS, definition.status_id, definition)
    state = BatchedFaultState(jax.random.PRNGKey(0), jnp.full((2, 2), 91), 91,
                              jnp.zeros((2, 2)))
    output, restored = jax.jit(apply_actuator_effects)((state,), jnp.ones((2, 2)), 0.)
    np.testing.assert_array_equal(output, np.full((2, 2), .25))
    assert restored[0].failure_value == 91
