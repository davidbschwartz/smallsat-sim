"""Configuration can be inspected and validated without allocating a simulator."""

import pytest

from smallsat_sim.envs.vehicles.astrobee_rl.config import resolve_config
from smallsat_sim.envs.vehicles.astrobee_rl.env import AstrobeeEnvVectorized
from smallsat_sim.api.experiments import ExperimentSpec, _astrobee_rl_config


def test_resolved_settings_are_independent():
    first = resolve_config()
    second = resolve_config()
    first.env.Bodies.bodies_list[0].pos[0] = 100
    first.training.PPO.steps_per_epoch = 1
    first.env.sim.seed = 123
    first.env.planner.bounds[0, 0] = -999
    assert second.env.Bodies.bodies_list[0].pos[0] == 0
    assert second.training.PPO.steps_per_epoch == 512
    assert second.env.sim.seed != 123
    assert second.env.planner.bounds[0, 0] == -10


def test_factory_and_direct_defaults_agree():
    direct = resolve_config()
    factory = _astrobee_rl_config(ExperimentSpec(vehicle="astrobee_rl"))
    assert direct.env.environment == factory.env.environment
    assert direct.training == factory.training
    assert direct.env.Bodies == factory.env.Bodies
    assert not _astrobee_rl_config(
        ExperimentSpec(vehicle="astrobee_rl", failures="off")
    ).env.environment.train_with_failures


def test_factory_validates_non_rl_settings_before_allocation():
    with pytest.raises(ValueError, match="Unknown override path"):
        _astrobee_rl_config(ExperimentSpec(
            vehicle="astrobee_rl", overrides={"PD.not_a_setting": 1}
        ))


def test_nested_and_dotted_overrides_resolve_identically():
    nested = resolve_config(overrides={"RL": {"num_envs": 4, "PPO": {"steps_per_epoch": 8}}})
    dotted = resolve_config(overrides={"num_envs": 4, "PPO.steps_per_epoch": 8})
    assert nested.env.environment == dotted.env.environment
    assert nested.training == dotted.training


def test_registered_rl_alias_uses_registration_defaults():
    from smallsat_sim.api.experiments import get_env, register_env, make_experiment

    register_env("test/rl_alias", get_env("astrobee_rl"), replace=True)
    experiment = make_experiment(
        vehicle="test/rl_alias", log=False, planner_radius=0,
        overrides={"RL": {
            "num_envs": 2, "use_adaptive_approach": False,
            "rollout_backend": "freeflyer",
            "PPO": {"steps_per_epoch": 2, "num_minibatches": 1},
        }},
    )
    try:
        assert experiment.runner is not None
        assert experiment.env.num_envs == 2
        assert experiment.run_spec.algorithm == "ppo"
    finally:
        experiment.env.close()


def test_domain_settings_and_old_inputs_resolve_identically():
    old = resolve_config(overrides={"RL": {
        "num_envs": 4, "algorithm": "vpg", "VPG": {"steps_per_epoch": 8},
    }})
    current = resolve_config(overrides={
        "environment": {"num_envs": 4},
        "training": {"algorithm": "vpg", "VPG.steps_per_epoch": 8},
    })
    factory = _astrobee_rl_config(ExperimentSpec(
        vehicle="astrobee_rl", overrides={
            "environment.num_envs": 4, "training.algorithm": "vpg",
            "training.VPG.steps_per_epoch": 8,
        },
    ))
    assert old.env.environment == current.env.environment == factory.env.environment
    assert old.training == current.training == factory.training
    assert not hasattr(current.env.control, "RL")
    assert "PPO" not in current.env.environment
    assert "reward" not in current.training
    assert "num_envs" not in current.training


def test_old_checkpoint_settings_roundtrip_through_input_translation():
    config = resolve_config(num_envs=3, algorithm="vpg")
    flat_checkpoint_settings = {**config.env.environment, **config.training}
    restored = resolve_config(overrides=flat_checkpoint_settings)
    assert restored.env.environment == config.env.environment
    assert restored.training == config.training


def test_old_and_new_aliases_cannot_silently_override_each_other():
    with pytest.raises(ValueError, match="Duplicate override path"):
        resolve_config(overrides={"RL.num_envs": 4, "environment.num_envs": 8})
    with pytest.raises(ValueError, match="Conflicting algorithm"):
        resolve_config(algorithm="ppo", overrides={"training.algorithm": "vpg"})


def test_resolver_returns_separate_environment_and_training_values():
    config = resolve_config(overrides={"training.context_window_len": 12})
    assert set(config) == {"env", "training"}
    assert "training" not in config.env
    assert config.env.context.history_len == 12
    config.training.context_scale[0] = 9
    assert config.env.context.scale[0] == 1
    assert resolve_config().training.context_scale[0] == 1


def test_mismatched_policy_context_fails_before_runner_allocation():
    from types import SimpleNamespace
    from smallsat_sim.controllers.rl.runners.on_policy_runner import OnPolicyRunner

    config = resolve_config()
    config.training.context_window_len += 1
    with pytest.raises(ValueError, match="Policy context differs at history_len"):
        OnPolicyRunner(SimpleNamespace(env_cfg=config.env), None, config=config.training)


@pytest.mark.parametrize("supplied_vehicle", [False, True])
def test_environment_snapshots_only_its_inputs(monkeypatch, supplied_vehicle):
    from smallsat_sim.envs.vehicles.astrobee_rl import env as env_module
    from smallsat_sim.model.vehicle import load_vehicle

    class SetupReached(Exception):
        pass

    def stop_before_simulator(self, args):
        raise SetupReached

    paths = []
    def load(path):
        paths.append(path)
        return load_vehicle(path)

    monkeypatch.setattr(env_module.VecEnv, "__init__", stop_before_simulator)
    monkeypatch.setattr(env_module, "load_vehicle", load)
    config = resolve_config(num_envs=2)
    physical = load_vehicle("vehicles/astrobee.yaml") if supplied_vehicle else None
    env = AstrobeeEnvVectorized.__new__(AstrobeeEnvVectorized)
    with pytest.raises(SetupReached):
        env.__init__(None, config=config.env, vehicle=physical)
    assert "training" not in env.env_cfg
    assert paths == ([] if supplied_vehicle else ["vehicles/astrobee.yaml"])
    if supplied_vehicle:
        assert env.model_cfg is physical
    config.env.Bodies.bodies_list[0].pos[0] = 999
    assert env.env_cfg.Bodies.bodies_list[0].pos[0] == 0
