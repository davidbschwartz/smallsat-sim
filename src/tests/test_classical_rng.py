"""Classical faults have reproducible, isolated random streams."""

from importlib import import_module
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from smallsat_sim.model.vehicle import load_vehicle
from smallsat_sim.envs.effects.classical import ThrusterFailureSimulator, PerturbationStatus


def test_config_import_does_not_initialize_global_rngs():
    subprocess.run([sys.executable, '-c', '''
import random, sys
import numpy as np
random.seed(987)
np.random.seed(654)
python_state, numpy_state = random.getstate(), np.random.get_state()
from smallsat_sim.envs.config import load_env_settings
load_env_settings("astrobee")
assert "torch" not in sys.modules
assert python_state == random.getstate()
after = np.random.get_state()
assert numpy_state[0] == after[0]
np.testing.assert_array_equal(numpy_state[1], after[1])
assert numpy_state[2:] == after[2:]
'''], check=True)


@pytest.mark.parametrize('name,cls,expected', [
    ('astrobee', 'AstrobeeEnv', ['StuckOffThrusters','StuckOnThrusters','FaultyValve','SaturatedThrust','ThrustInstability']),
    ('astrobee_benchmark', 'AstrobeeBenchmarkEnv', ['StuckOffThrusters','StuckOnThrusters','FaultyValve','SaturatedThrust','ThrustInstability']),
])
def test_fault_reset_preserves_order_and_rng_isolation(name, cls, expected):
    env_type = getattr(import_module(f'smallsat_sim.envs.vehicles.{name}.env'), cls)
    def make():
        env = object.__new__(env_type)
        env.model_cfg = load_vehicle('vehicles/astrobee.yaml')
        env.env_cfg = SimpleNamespace(sim=SimpleNamespace(verbose=False))
        env.np_rng = np.random.RandomState(7)
        env.reset_perturbations()
        return env
    first, second = make(), make()
    assert [type(p).__name__ for p in first.perturbations.perturbations] == expected
    expected_indices = [second.perturbations.perturbations[0].select_thruster(None) for _ in range(6)]
    make().np_rng.normal(size=100)
    actual = [first.perturbations.perturbations[0].select_thruster(None) for _ in range(6)]
    np.testing.assert_array_equal(actual, expected_indices)
    first.perturbations.perturbations[0].stuck_off_thruster(0, 0.)
    first.np_rng.seed(7)
    first.reset_perturbations()
    assert all(p.rng is first.np_rng for p in first.perturbations.perturbations)
    assert np.all(np.asarray(first.perturbations.perturbations[0].thruster_mask) == 0)
    np.testing.assert_array_equal(
        [first.perturbations.perturbations[0].select_thruster(None) for _ in range(6)], expected_indices,
    )


@pytest.mark.parametrize('kind', [PerturbationStatus.FAULTY_VALVE, PerturbationStatus.SATURATED_THRUST, PerturbationStatus.THRUST_INSTABILITY])
def test_gp_faults_are_seeded_without_changing_torch_rng(kind):
    before = torch.random.get_rng_state().clone()
    def sample(seed):
        return ThrusterFailureSimulator(rng=np.random.RandomState(seed)).generate_failure_data(kind)[1]
    first = sample(7)
    sample(19)
    torch.testing.assert_close(first, sample(7), rtol=0, atol=0)
    assert torch.equal(before, torch.random.get_rng_state())
