"""Astrobee RL failure-state reset behavior tests."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp

from smallsat_sim.envs.vehicles.astrobee_rl.env import AstrobeeEnvVectorized
from smallsat_sim.envs.effects.preparation import (
    ThrusterOccupancy, StuckOffThrusters,
)
from smallsat_sim.envs.effects.actuator_kernels import (
    PerturbationStatus,
)


def _make_dummy_env_cfg(num_envs: int):
    return SimpleNamespace(
        environment=SimpleNamespace(num_envs=num_envs),
        sim=SimpleNamespace(verbose=False),
    )


def _make_dummy_model_cfg(num_thrusters: int):
    thrusters = [
        SimpleNamespace(forcerange=(0.0, 1.0), ctrlrange=(0.0, 1.0))
        for _ in range(num_thrusters)
    ]
    return SimpleNamespace(
        actuators=thrusters
    )


def test_reset_perturbations_clears_environment_occupancy() -> None:
    num_envs = 3
    num_thrusters = 4

    class _FakeEnv:
        fault_specs = ()
        _schedule_configured_faults = AstrobeeEnvVectorized._schedule_configured_faults

        def __init__(self):
            self.num_envs = num_envs
            self.thruster_occupancy = ThrusterOccupancy(num_envs, num_thrusters)
            self.env_cfg = _make_dummy_env_cfg(num_envs)
            self.model_cfg = _make_dummy_model_cfg(num_thrusters)
            effect = StuckOffThrusters(self.env_cfg, self.model_cfg, jax.random.PRNGKey(1),
                                      occupancy=self.thruster_occupancy)
            effect.activate(jax.random.PRNGKey(2), jnp.arange(num_envs), 0)

        def next_rng_keys(self, count: int):
            return jax.random.split(jax.random.PRNGKey(0), count)

        def _refresh_effect_states(self) -> None:
            pass

    fake_env = _FakeEnv()

    AstrobeeEnvVectorized.reset_perturbations(fake_env)

    assert fake_env.thruster_occupancy.mask.shape == (num_envs, num_thrusters)
    assert jnp.all(fake_env.thruster_occupancy.mask == PerturbationStatus.OPERATIONAL.value)
