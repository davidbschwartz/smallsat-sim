"""Experiment-builder registry and construction tests."""

from pathlib import Path

import pytest

from smallsat_sim.api import RegistryError
from smallsat_sim.api.experiments import (
    register_controller,
    make_experiment,
    make_rl_experiment,
)
from smallsat_sim.controllers.pd.controller import PDController
from smallsat_sim.envs.vehicles.astrobee_rl import config as rl_config


def test_make_rl_experiment_builds_tiny_astrobee_rl_stack(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")

    original_num_envs = rl_config.resolve_config().env.environment.num_envs
    env = None
    try:
        env, planner, runner = make_rl_experiment(
            vehicle="astrobee_rl",
            reward="vec_env/full_pose",
            failures="off",
            run_name="builder_smoke",
            planner_radius=0.0,
            overrides={
                "RL.num_envs": 4,
                "RL.PPO.steps_per_epoch": 4,
                "RL.PPO.epochs": 1,
                "RL.PPO.max_ep_len": 4,
                "RL.episode_len": 4,
                "RL.n_evals": 1,
            },
        )
        runner.ckpt_dir = str(tmp_path) + "/"

        assert env.num_envs == 4
        assert env.env_cfg.environment.reward == "vec_env/full_pose"
        assert not env.train_with_failures
        assert env.run_spec.reward == "vec_env/full_pose"
        assert runner.run_spec is env.run_spec
        assert planner.reference_points.shape[0] == 1
    finally:
        if env is not None:
            env.close()
        assert rl_config.resolve_config().env.environment.num_envs == original_num_envs


def test_make_rl_experiment_rejects_unknown_vehicle() -> None:
    with pytest.raises(RegistryError, match="Unknown env"):
        make_rl_experiment(vehicle="not_registered")


@pytest.mark.parametrize("vehicle", ["astrobee", "cubesat", "sprint"])
def test_make_experiment_builds_classical_pd_stack(vehicle):
    experiment = make_experiment(
        vehicle=vehicle, controller="pd", planner="oracle",
        planner_radius=0.0, overrides={"PD.gains.Kp_x": 0.25}, log=False,
    )
    try:
        assert isinstance(experiment.controller, PDController)
        assert experiment.runner is None
        assert experiment.env.run_spec is experiment.run_spec
        assert experiment.controller.Kp_x == 0.25
        assert experiment.run_spec.vehicle == vehicle
        assert experiment.env.model_cfg.name == vehicle
        assert experiment.run_spec.algorithm == "pd"
    finally:
        experiment.env.close()


def test_make_experiment_accepts_registered_controller() -> None:
    pytest.importorskip("mujoco")

    class DummyController:
        def __init__(self, env, planner, run_spec) -> None:
            self.env = env
            self.planner = planner
            self.run_spec = run_spec

    @register_controller("test/dummy_controller", replace=True)
    def build_dummy_controller(env, planner, spec, run_spec, training_config):
        assert spec.controller == "test/dummy_controller"
        return DummyController(env, planner, run_spec)

    experiment = make_experiment(
        vehicle="astrobee",
        controller="test/dummy_controller",
        planner="oracle",
        planner_radius=0.0,
        log=False,
    )

    try:
        assert isinstance(experiment.controller, DummyController)
        assert experiment.controller.env is experiment.env
        assert experiment.controller.run_spec is experiment.run_spec
    finally:
        experiment.env.close()


def test_make_experiment_rejects_unknown_env_before_construction() -> None:
    with pytest.raises(RegistryError, match="Unknown env"):
        make_experiment(vehicle="astrobee", env="not_registered")


def test_make_experiment_rejects_unknown_planner_before_construction() -> None:
    with pytest.raises(RegistryError, match="Unknown planner"):
        make_experiment(vehicle="astrobee", planner="not_registered")


def test_make_experiment_rejects_unknown_controller() -> None:
    with pytest.raises(RegistryError, match="Unknown controller"):
        make_experiment(
            vehicle="astrobee",
            controller="not_registered",
            planner="oracle",
            planner_radius=0.0,
            log=False,
        )


def test_registered_environment_resolves_its_asset_once(monkeypatch):
    from smallsat_sim.api import experiments
    from smallsat_sim.envs import base_env
    from smallsat_sim.envs.vehicles.astrobee.env import AstrobeeEnv

    experiments.register_env(
        "test/physical_asset",
        experiments.EnvEntry(AstrobeeEnv, "vehicles/astrobee.yaml"),
        replace=True,
    )
    original_load = experiments.load_vehicle
    loaded = []

    def load(path):
        definition = original_load(path)
        loaded.append((path, definition))
        return definition

    def unexpected_default_load(path):
        raise AssertionError("The environment must use the already selected asset")

    monkeypatch.setattr(experiments, "load_vehicle", load)
    monkeypatch.setattr(base_env, "load_vehicle", unexpected_default_load)
    experiment = make_experiment(
        vehicle="test/physical_asset", planner_radius=0.0, log=False,
    )
    try:
        assert len(loaded) == 1
        assert loaded[0][0] == "vehicles/astrobee.yaml"
        assert experiment.env.model_cfg is loaded[0][1]
    finally:
        experiment.env.close()
