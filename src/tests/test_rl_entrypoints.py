"""Experiment workflows retain their stages, trajectory and output controls."""
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.rl_training import train_astrobee
from experiments import astrobee_RL


def args(**changes):
    return SimpleNamespace(**{
        "headless": True, "viewer": False, "video": False,
        "log": False, "wandb": False, "checkpoint": None, **changes,
    })


@pytest.mark.parametrize("adaptive", [False, True])
def test_training_workflow_keeps_stages_and_config_defaults(monkeypatch, adaptive):
    events = []
    options = args(log=True)
    env = SimpleNamespace(
        use_adaptive_approach=adaptive,
        logger=SimpleNamespace(save_log=lambda: events.append("log")),
        close=lambda: events.append("close"),
    )

    def build_env(**kwargs):
        assert kwargs["args"] is options
        assert "training" not in kwargs["config"]
        return env

    def planner(actual_env, *, radius):
        assert actual_env is env and radius == 0.0
        return "planner"

    def runner(actual_env, actual_planner, *, config):
        assert actual_env is env and actual_planner == "planner"
        return SimpleNamespace(**{
            name: lambda name=name: events.append(name)
            for name in ("pretrain", "learn", "train_adaptation_module_on_policy", "evaluate")
        })

    monkeypatch.setattr(train_astrobee, "get_args", lambda: options)
    monkeypatch.setattr(train_astrobee, "AstrobeeEnvVectorized", build_env)
    monkeypatch.setattr(train_astrobee, "OraclePlannerRL", planner)
    monkeypatch.setattr(train_astrobee, "make_runner", runner)
    train_astrobee.main()
    expected = ["pretrain", "learn"]
    if adaptive:
        expected.append("train_adaptation_module_on_policy")
    assert events == [*expected, "evaluate", "log", "close"]


def test_training_closes_environment_if_setup_fails(monkeypatch):
    closed = []
    monkeypatch.setattr(train_astrobee, "get_args", args)
    monkeypatch.setattr(train_astrobee, "AstrobeeEnvVectorized",
                        lambda **kwargs: SimpleNamespace(close=lambda: closed.append(True)))

    def fail(*args, **kwargs):
        raise RuntimeError("planner failed")

    monkeypatch.setattr(train_astrobee, "OraclePlannerRL", fail)
    with pytest.raises(RuntimeError, match="planner failed"):
        train_astrobee.main()
    assert closed == [True]


@pytest.mark.parametrize("with_checkpoint", [False, True])
def test_deployment_preserves_path_and_start_position(monkeypatch, tmp_path, with_checkpoint):
    events = []
    checkpoint = tmp_path / "renamed-policy.ckpt" if with_checkpoint else None
    options = args(checkpoint=checkpoint, log=True)
    settings = {"algorithm": "ppo", "deployment_len": 2}
    monkeypatch.setattr(astrobee_RL, "load_trained_modules", lambda *args: {
        "metadata": {"experiment": "saved", "resolved_config": {"seed": 9, "rl": settings}},
    })
    env = SimpleNamespace(close=lambda: events.append("close"),
                          logger=SimpleNamespace(save_log=lambda: events.append("log")))

    def build_env(**kwargs):
        assert kwargs["args"] is options
        np.testing.assert_allclose(kwargs["config"].Bodies.bodies_list[0].pos, [5., 0., 10.17])
        assert kwargs["config"].Bodies.max_start_offset == .5
        assert kwargs["run_name"] == ("saved" if with_checkpoint else "default")
        return env

    def planner(actual_env):  # No radius=0 benchmark override.
        assert actual_env is env
        return "circular planner"

    controller = SimpleNamespace(
        ckpt_filename="policy.ckpt", adaptation_module_file_name="am.ckpt",
        control=lambda: events.append("control"),
    )
    monkeypatch.setattr(astrobee_RL, "AstrobeeEnvVectorized", build_env)
    monkeypatch.setattr(astrobee_RL, "OraclePlannerRL", planner)
    def build_controller(actual_env, actual_planner, *, config, checkpoint):
        assert checkpoint == options.checkpoint
        return controller

    monkeypatch.setattr(astrobee_RL, "RLController", build_controller)
    astrobee_RL.run(options)
    assert events == ["control", "log", "close"]
