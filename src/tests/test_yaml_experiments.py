"""Exercise the public YAML path through real asset, reward and environment creation."""

from smallsat_sim.configuration import apply_overrides, rl_overrides, settings

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from smallsat_sim.api import load_vehicle, register_reward, register_termination
from smallsat_sim.envs.termination import TerminationResult
from smallsat_sim.api.experiments import make_experiment
from smallsat_sim.api.vec_env_rewards import full_pose_reward
from smallsat_sim.configuration import load_settings, read_yaml


def test_yaml_array_and_diagonal_settings(tmp_path):
    path = tmp_path / "settings.yaml"
    path.write_text("cost:\n  Q: !diag [1.0, 2.0]\n  R: !array [[3.0, 4.0]]\n")
    cfg = load_settings(path)
    np.testing.assert_array_equal(cfg.cost.Q, np.diag([1.0, 2.0]))
    np.testing.assert_array_equal(cfg.cost.R, [[3.0, 4.0]])


@pytest.mark.parametrize("backend", ["freeflyer", "mjx"])
def test_yaml_asset_and_python_reward_reach_the_simulator(tmp_path, backend):
    asset = read_yaml("vehicles/astrobee.yaml")
    asset["name"] = "yaml_test_satellite"
    asset["physical"]["mass"] = 12.0
    asset_path = tmp_path / "satellite.yaml"
    asset_path.write_text(yaml.safe_dump(asset))
    assert load_vehicle(asset_path).physical.mass == 12.0

    @register_reward("yaml_test_reward", replace=True)
    def reward(context):
        result = full_pose_reward(context)
        return replace(
            result,
            rewards=jnp.full_like(result.rewards, 7.0),
        )

    @register_termination("yaml_test_termination", replace=True)
    def termination(context):
        success = jnp.arange(context.next_states.shape[0]) == 0
        return TerminationResult(
            success, success, jnp.zeros_like(success), context.terminal_hold_counts + 3,
        )

    path = tmp_path / "experiment.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "vehicle": "astrobee_rl",
                "asset": asset_path.name,
                "reward": "yaml_test_reward",
                "planner_radius": 0.0,
                "headless": True,
                "log": False,
                "overrides": {
                    "RL": {
                        "use_adaptive_approach": False,
                        "num_envs": 2,
                        "rollout_backend": backend,
                        "termination": "yaml_test_termination",
                        "PPO": {"steps_per_epoch": 2, "num_minibatches": 1},
                    }
                },
            }
        )
    )
    experiment = make_experiment(path)
    try:
        assert experiment.env.model_cfg.physical.mass == 12.0
        assert experiment.env.model.body_mass[1] == 12.0
        assert experiment.runner.am is None
        collector = experiment.runner.collector("zero", stochastic=False)
        result = experiment.runner.collect(collector, randomize=False)
        np.testing.assert_array_equal(result.step_outputs.rewards, np.full((2, 2), 7.0))
        np.testing.assert_array_equal(result.final_state.terminal_hold_counts, [3, 6])
        np.testing.assert_array_equal(
            result.terminated_masks, [[True, False], [True, False]]
        )
    finally:
        experiment.env.close()


def test_invalid_yaml_override_fails_before_running():
    with pytest.raises(ValueError, match="Unknown override path"):
        make_experiment(
            "experiments/ppo_nominal.yaml", overrides={"RL.not_a_setting": 1}
        )


def test_vpg_override_is_preserved_by_experiment_builder():
    experiment = make_experiment(
        vehicle="astrobee_rl",
        log=False,
        planner_radius=0.0,
        overrides={
            "RL": {
                "algorithm": "vpg",
                "use_adaptive_approach": False,
                "num_envs": 2,
                "rollout_backend": "freeflyer",
                "VPG": {"steps_per_epoch": 2, "num_minibatches": 1},
            }
        },
    )
    try:
        assert experiment.runner.agent.algorithm == "vpg"
        assert experiment.run_spec.algorithm == "vpg"
        assert experiment.runner.resolved_config["rl"]["algorithm"] == "vpg"
    finally:
        experiment.env.close()


def test_conflicting_algorithm_is_rejected():
    with pytest.raises(ValueError, match="Conflicting algorithm"):
        make_experiment(
            vehicle="astrobee_rl",
            algorithm="ppo",
            log=False,
            overrides={"RL.algorithm": "vpg"},
        )


def test_benchmark_and_api_share_override_paths():
    values = {"RL": {"algorithm": "vpg", "VPG.steps_per_epoch": 8}}
    control = settings({"RL": {"algorithm": "ppo", "VPG": {"steps_per_epoch": 4}}})
    benchmark = settings({"algorithm": "ppo", "VPG": {"steps_per_epoch": 4}})
    apply_overrides(control, values)
    apply_overrides(benchmark, rl_overrides(values))
    assert control.RL == benchmark
    with pytest.raises(ValueError, match="Duplicate override"):
        rl_overrides({"RL": {"algorithm": "ppo"}, "RL.algorithm": "vpg"})
