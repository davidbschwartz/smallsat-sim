"""Fault schedules reach the real actuator state and retain onset/reset behavior."""

from dataclasses import asdict

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from smallsat_sim.api import FaultSpec, validate_fault
from smallsat_sim.api.experiments import make_experiment
from smallsat_sim.envs.effects.catalog import configured_faults
from smallsat_sim.model.vehicle import load_vehicle


def test_fault_validation_resolves_names_and_rejects_ambiguous_targets():
    vehicle = load_vehicle("vehicles/astrobee.yaml")
    assert validate_fault(FaultSpec("stuck_off", vehicle.actuators[0].name), vehicle, 2) == 0
    for spec, message in [
        (FaultSpec("identity"), "Unknown fault"),
        (FaultSpec("stuck_off", "typo"), "Unknown actuator"),
        (FaultSpec("stuck_on", start_time=-1), "start_time"),
        (FaultSpec("stuck_on", env_ids=(2,)), "env_ids"),
        (FaultSpec("faulty_valve", minimum=.2), "both"),
    ]:
        with pytest.raises(ValueError, match=message):
            validate_fault(spec, vehicle, 2)
    spec = FaultSpec("stuck_off", vehicle.actuators[0].name)
    with pytest.raises(ValueError, match="only one"):
        configured_faults([spec, spec], vehicle, 2)


@pytest.mark.parametrize("backend", ["freeflyer", "mjx"])
def test_configured_fault_reaches_rollout_and_is_reapplied_on_reset(backend):
    name = load_vehicle("vehicles/astrobee.yaml").actuators[0].name
    spec = FaultSpec("stuck_off", actuator=name, env_ids=(0,))
    experiment = make_experiment(
        vehicle="astrobee_rl", log=False, planner_radius=0, failures="off",
        overrides={"RL": {
            "num_envs": 2, "use_adaptive_approach": False,
            "rollout_backend": backend, "faults": [asdict(spec)],
            "PPO": {"steps_per_epoch": 2, "num_minibatches": 1},
        }},
    )
    try:
        env = experiment.env
        controls = jnp.ones((2, 12)) * .2
        before = env.perturbations.apply(controls, 0)
        assert float(before[0, 0]) == 0
        assert float(before[1, 0]) > 0
        env.reset_perturbations()
        np.testing.assert_array_equal(env.perturbations.apply(controls, 0), before)
        result = experiment.runner.collect(experiment.runner.collector("zero", stochastic=False), randomize=False)
        np.testing.assert_array_equal(result.step_outputs.applied_ctrl[:, 0, 0], [0, 0])
        assert experiment.runner.resolved_config["rl"]["faults"][0]["effect"] == "stuck_off"
        with pytest.raises(ValueError, match="already"):
            env.schedule_fault(spec, key=jax.random.PRNGKey(1))
        env.schedule_fault(FaultSpec("stuck_off", name, start_time=2, env_ids=(1,)), key=jax.random.PRNGKey(2))
        assert float(env.perturbations.apply(controls, 1)[1, 0]) > 0
        assert float(env.perturbations.apply(controls, 2)[1, 0]) == 0
    finally:
        experiment.env.close()


def test_gp_fault_scheduling_passes_bounds_and_retains_onset():
    experiment = make_experiment(
        vehicle="astrobee_rl", log=False, planner_radius=0, failures="off",
        overrides={"RL": {"num_envs": 2, "use_adaptive_approach": False,
            "rollout_backend": "freeflyer", "PPO": {"steps_per_epoch": 2, "num_minibatches": 1}}},
    )
    try:
        env = experiment.env
        actuator = env.model_cfg.actuators[0]
        upper = actuator.ctrlrange[-1]
        for effect_index, name in enumerate(("faulty_valve", "saturated_thrust", "thrust_instability"), 2):
            env.reset_perturbations()
            effect = env.perturbations.perturbations[effect_index]
            original = effect._gp_simulator_kwargs
            received = []
            def capture(**kwargs):
                received.append(kwargs)
                return original(**kwargs)
            effect._gp_simulator_kwargs = capture
            spec = FaultSpec(name, actuator.name, start_time=2, env_ids=(0,), minimum=.2*upper, maximum=.7*upper)
            env.schedule_fault(spec, key=jax.random.PRNGKey(effect_index))
            assert float(received[0]["valve_min"]) == pytest.approx(.2*upper)
            assert float(received[0]["valve_max"]) == pytest.approx(.7*upper)
            assert float(effect.state.start_times[0, 0]) == 2
            controls = jnp.ones((2, len(env.model_cfg.actuators))) * .1
            np.testing.assert_array_equal(effect.apply(controls, 1), controls)
            assert effect.state.gp_x_samples is not None
    finally:
        experiment.env.close()


def test_environment_occupancy_is_isolated_and_restored_with_snapshot():
    def make(num_envs):
        return make_experiment(
            vehicle="astrobee_rl", log=False, planner_radius=0, failures="off",
            overrides={"RL": {"num_envs": num_envs, "use_adaptive_approach": False,
                "rollout_backend": "freeflyer",
                "PPO": {"steps_per_epoch": 2, "num_minibatches": 1}}},
        )

    first = make(2)
    second = None
    try:
        env = first.env
        names = [actuator.name for actuator in env.model_cfg.actuators]
        spec = FaultSpec("stuck_on", names[0], start_time=2, env_ids=(0,))
        env.schedule_fault(spec, key=jax.random.PRNGKey(10))
        saved = env.state_struct
        expected_mask = np.asarray(env.thruster_occupancy.mask).copy()
        controls = jnp.ones((2, len(names))) * .1
        expected = env.perturbations.apply(controls, 3)

        second = make(3)
        second.env.schedule_fault(
            FaultSpec("stuck_off", names[0]), key=jax.random.PRNGKey(11),
        )
        second.env.reset_perturbations()
        np.testing.assert_array_equal(env.thruster_occupancy.mask, expected_mask)
        np.testing.assert_array_equal(env.perturbations.apply(controls, 3), expected)
        assert env.thruster_occupancy is not second.env.thruster_occupancy
        assert all(effect.occupancy is env.thruster_occupancy
                   for effect in env.perturbations.perturbations)
        with pytest.raises(ValueError, match="already"):
            env.schedule_fault(spec, key=jax.random.PRNGKey(12))

        env.reset_perturbations()
        env.apply_state_struct(saved)
        np.testing.assert_array_equal(env.thruster_occupancy.mask, expected_mask)
        np.testing.assert_array_equal(env.perturbations.apply(controls, 3), expected)
        restored = env.perturbations.perturbations[1]
        np.testing.assert_array_equal(restored._key, saved.perturbation_states[1].rng)
        np.testing.assert_array_equal(restored.start_times, saved.perturbation_states[1].start_times)
        np.testing.assert_array_equal(restored.stuck_on_force, saved.perturbation_states[1].max_thruster_force)
        with pytest.raises(ValueError, match="already"):
            env.schedule_fault(spec, key=jax.random.PRNGKey(13))
        env.schedule_fault(
            FaultSpec("stuck_on", names[1], start_time=4, env_ids=(1,)),
            key=jax.random.PRNGKey(14),
        )
        assert float(restored.state.start_times[0, 0]) == 2
        np.testing.assert_array_equal(env.perturbations.apply(controls, 1), controls)
    finally:
        first.env.close()
        if second is not None:
            second.env.close()
