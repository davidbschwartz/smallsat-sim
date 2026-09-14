"""Guard tests for mutually exclusive shared-thruster GP faults."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp

from smallsat_sim.envs.effects.preparation import (
    FaultyValve, StuckOffThrusters, StuckOnThrusters,
)
from smallsat_sim.envs.effects.actuator_kernels import (
    PerturbationStatus,
)


def _make_env_cfg(num_envs: int):
    return SimpleNamespace(
        environment=SimpleNamespace(num_envs=num_envs),
        sim=SimpleNamespace(verbose=False),
    )


def _make_model_cfg(num_thrusters: int):
    thruster_list = [
        SimpleNamespace(ctrlrange=(0.0, 1.0), forcerange=(0.0, 1.0))
        for _ in range(num_thrusters)
    ]
    return SimpleNamespace(
        actuators=thruster_list
    )


def test_gp_registration_skips_when_no_shared_operational_thruster() -> None:
    env_cfg = _make_env_cfg(num_envs=2)
    model_cfg = _make_model_cfg(num_thrusters=2)
    gp = FaultyValve(env_cfg, model_cfg, key=jax.random.PRNGKey(0))

    # Make env 0 only thruster 1 operational, env 1 only thruster 0 operational.
    off = StuckOffThrusters(env_cfg, model_cfg, jax.random.PRNGKey(2), occupancy=gp.occupancy)
    on = StuckOnThrusters(env_cfg, model_cfg, jax.random.PRNGKey(3), occupancy=gp.occupancy)
    off.activate(jax.random.PRNGKey(4), jnp.array([0]), 0)
    on.activate(jax.random.PRNGKey(5), jnp.array([1]), 1)
    before = gp.occupancy.mask

    gp.register_perturbation(
        key=jax.random.PRNGKey(1),
        perturbed_envs=jnp.array([0, 1], dtype=jnp.int32),
    )

    assert jnp.array_equal(gp.occupancy.mask, before)
