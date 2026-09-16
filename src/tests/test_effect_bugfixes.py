"""Regression tests for effect routing and serialization bug fixes."""

from types import SimpleNamespace
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from smallsat_sim.envs.effects import preparation
from smallsat_sim.envs.effects.custom import CustomEffect, CustomEffectState


def setup():
    return (SimpleNamespace(environment=SimpleNamespace(num_envs=2), sim=SimpleNamespace(verbose=False)),
            SimpleNamespace(actuators=[SimpleNamespace(forcerange=(0., 1.), ctrlrange=(0., 1.))] * 2))


@pytest.mark.parametrize('kind', [preparation.StuckOffThrusters, preparation.StuckOnThrusters])
def test_exhausted_row_does_not_overwrite_last_actuator(kind):
    cfg, vehicle = setup()
    effect = kind(cfg, vehicle, jax.random.PRNGKey(0))
    effect.activate(jax.random.PRNGKey(1), jnp.array([0]), 0)
    effect.activate(jax.random.PRNGKey(2), jnp.array([0]), 1)
    before = effect.state
    effect.activate(jax.random.PRNGKey(3), jnp.array([0]), start_time=10.)
    np.testing.assert_array_equal(effect.state.start_times, before.start_times)
    if before.max_thruster_force is not None:
        np.testing.assert_array_equal(effect.state.max_thruster_force, before.max_thruster_force)


def test_custom_wrench_keyword_time_and_explicit_method():
    effect = CustomEffect.__new__(CustomEffect)
    effect.disturbance = True
    effect.value_shape = (1, 6)
    effect.state = CustomEffectState({}, jnp.array([True]), jnp.array([2.]),
                                    'test:apply', .1, lambda data, values, time, dt: (values + time, data))
    np.testing.assert_array_equal(effect.apply(timestamp=3.), np.full((1, 6), 3.))
    np.testing.assert_array_equal(effect.apply(3.), effect.apply_wrench(3.))


def test_gp_settings_change_invalidates_curve_and_failure_does_not_install_targets(monkeypatch):
    cfg, vehicle = setup()
    calls = []
    class Sampler:
        def __init__(self, **kwargs):
            calls.append(kwargs)
        def generate_failure_data(self, key, mode):
            return jnp.array([0., 1.]), jnp.array([0., .5])
    monkeypatch.setattr(preparation, 'ThrusterFailureSimulator', Sampler)
    effect = preparation.FaultyValve(cfg, vehicle, jax.random.PRNGKey(0))
    def activate(row):
        effect.activate(jax.random.PRNGKey(1), jnp.array([row]), 0, valve_min=.2, valve_max=.8)
    activate(0)
    saved = effect.state
    activate(1)
    assert len(calls) == 1
    effect.restore_state(saved)
    effect._gp_failure_mode_overrides = {'FAULTY_VALVE': {'lengthscale': .6}}
    activate(1)
    assert len(calls) == 2
    effect.restore_state(saved)
    monkeypatch.setattr(preparation, 'gp_register_state', lambda *args: (_ for _ in ()).throw(ValueError('table failure')))
    with pytest.raises(ValueError, match='table failure'):
        activate(1)
    np.testing.assert_array_equal(effect.state.thruster_mask, saved.thruster_mask)


@pytest.mark.parametrize('first', ['catalog', 'actuator_kernels'])
def test_import_order_does_not_load_host_samplers(first):
    subprocess.run([sys.executable, '-c', f'''
import importlib, sys
importlib.import_module('smallsat_sim.envs.effects.{first}')
from smallsat_sim.envs.effects import actuator_kernels, catalog
assert 'smallsat_sim.envs.effects.gp' not in sys.modules
assert 'smallsat_sim.envs.effects.preparation' not in sys.modules
assert 'smallsat_sim.envs.effects.classical' not in sys.modules
assert 'torch' not in sys.modules
'''], check=True)
