"""Environment settings resolve before allocation and never share mutable defaults."""

from functools import partial

import numpy as np
import pytest
import yaml

from smallsat_sim.api.experiments import EnvEntry, make_experiment, register_env
from smallsat_sim.api.registry import RegistryError
from smallsat_sim.envs.config import randomize_initial_pose
from smallsat_sim.configuration import settings


@pytest.mark.parametrize("name", [
    "astrobee", "cubesat", ])
def test_resolver_returns_independent_seeded_settings(name):
    from smallsat_sim.envs.config import resolve_env_config
    resolve = partial(resolve_env_config, name)
    first, second = resolve(seed=7), resolve(seed=7)
    assert first.Bodies == second.Bodies
    first.Bodies.bodies_list[0].pos[0] = 999
    first.sim.dt = .5
    assert second.Bodies.bodies_list[0].pos[0] != 999
    assert second.sim.dt != .5
    if hasattr(first, "planner"):
        first.planner.bounds[0, 0] = -999
        assert second.planner.bounds[0, 0] != -999


def test_pose_sampling_uses_configured_seed_without_consuming_global_rng():
    from smallsat_sim.envs.config import resolve_astrobee_config as resolve_config

    before = np.random.get_state()
    first, second = resolve_config(seed=7), resolve_config(seed=8)
    after = np.random.get_state()
    assert first.Bodies.bodies_list[0].pos != second.Bodies.bodies_list[0].pos
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_pose_sampling_does_not_mutate_source_attitude():
    body = settings({"pos": [0, 0, 0], "euler": np.zeros(3)})
    config = settings({"position_noise": 0, "attitude_noise_degrees": [1, 2]})
    _, attitude = randomize_initial_pose(body, config, rng=np.random.RandomState(1))
    assert attitude == [1, 1, 1]
    np.testing.assert_array_equal(body.euler, np.zeros(3))


@pytest.mark.parametrize("use_yaml", [False, True])
def test_factory_applies_structural_overrides_before_compiling(tmp_path, use_yaml):
    values = {
        "vehicle": "cubesat", "headless": True, "log": False,
        "planner_radius": 0,
        "overrides": {"sim.dt": .025, "PD.gains.Kp_x": .35},
    }
    if use_yaml:
        path = tmp_path / "experiment.yaml"
        path.write_text(yaml.safe_dump(values))
        experiment = make_experiment(path)
    else:
        experiment = make_experiment(**values)
    try:
        assert experiment.env.env_cfg.sim.dt == .025
        assert experiment.env.model.opt.timestep == .025
        assert experiment.controller.Kp_x == .35
    finally:
        experiment.env.close()


def test_unknown_override_fails_before_asset_loading(monkeypatch):
    from smallsat_sim.api import experiments

    def unexpected_load(path):
        raise AssertionError("Invalid configuration must fail before asset loading")
    monkeypatch.setattr(experiments, "load_vehicle", unexpected_load)
    with pytest.raises(ValueError, match="Unknown override path"):
        make_experiment(vehicle="astrobee", overrides={"PD.gains.typo": 1})


def test_custom_environment_needs_resolver_for_overrides():
    class NeverConstructed:
        def __init__(self, **kwargs):
            raise AssertionError("Overrides must not be applied after construction")
    register_env("test/no_resolver", EnvEntry(NeverConstructed, "vehicles/astrobee.yaml"), replace=True)
    with pytest.raises(RegistryError, match="config_builder"):
        make_experiment(vehicle="test/no_resolver", overrides={"sim.dt": .02})


def test_old_and_new_controller_override_aliases_cannot_collide():
    from smallsat_sim.envs.config import resolve_cubesat_config as resolve_config
    with pytest.raises(ValueError, match="Duplicate override"):
        resolve_config(overrides={"PD.gains.Kp_x": .2, "control.PD.gains.Kp_x": .3})


def test_supplied_configuration_is_snapshotted_without_resolving_again(monkeypatch):
    from smallsat_sim.envs import config as module
    from smallsat_sim.envs.base_env import BaseEnv
    from smallsat_sim.model.vehicle import load_vehicle

    config = module.resolve_env_config("astrobee", seed=7)
    class Loader:
        _load_cfg = BaseEnv._load_cfg
    def unexpected_resolve():
        raise AssertionError("Supplied settings must not resolve a second time")
    monkeypatch.setattr("smallsat_sim.envs.base_env.resolve_env_config", unexpected_resolve)
    loaded, _ = Loader()._load_cfg(
        "astrobee", config=config, vehicle=load_vehicle("vehicles/astrobee.yaml"),
    )
    config.sim.dt = 99
    config.Bodies.bodies_list[0].pos[0] = 999
    assert loaded.sim.dt != 99
    assert loaded.Bodies.bodies_list[0].pos[0] != 999


def test_benchmark_uses_training_scene_with_astrobee_faults():
    from argparse import Namespace
    from smallsat_sim.envs.vehicles.astrobee.env import AstrobeeEnv
    from smallsat_sim.envs.vehicles.astrobee_benchmark.env import AstrobeeBenchmarkEnv
    from smallsat_sim.envs.config import resolve_env_config

    config = resolve_env_config('astrobee', seed=7)
    config.viewer.use_viser = False
    args = Namespace(headless=True, video=False, log=False)
    classical = AstrobeeEnv(args, config=config)
    benchmark = AstrobeeBenchmarkEnv(args, config=config)
    try:
        assert benchmark.model.ngeom < classical.model.ngeom
        np.testing.assert_array_equal(
            benchmark.model.body_mass[benchmark.model.jnt_bodyid],
            classical.model.body_mass[classical.model.jnt_bodyid],
        )
        assert [type(fault) for fault in benchmark.perturbations.perturbations] == [
            type(fault) for fault in classical.perturbations.perturbations
        ]
        # Scene compilers have different Euler sequences; preserve each scene's
        # existing attitude interpretation while sharing the sampled position.
        np.testing.assert_array_equal(benchmark.data.qpos[:3], classical.data.qpos[:3])
        assert benchmark.env_cfg.Bodies == classical.env_cfg.Bodies
    finally:
        benchmark.close()
        classical.close()
