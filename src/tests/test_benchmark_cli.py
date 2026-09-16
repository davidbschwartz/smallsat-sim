"""Benchmark commands preserve seeds, checkpoint selection and cleanup."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np

import pytest

from experiments.rl_benchmarking import benchmark


def test_commands_require_their_own_inputs():
    parser = benchmark.build_parser()
    for command in ([], ["train"], ["evaluate"], ["deploy"],
                    ["evaluate", "--checkpoint", "policy.ckpt", "--config", "override.yaml"]):
        with pytest.raises(SystemExit):
            parser.parse_args(command)
    train = parser.parse_args(["train", "--experiment", "ppo_nominal", "--seed", "7"])
    evaluate = parser.parse_args(["evaluate", "--checkpoint", "policy.ckpt", "--seed", "19"])
    assert train.seed == 7 and evaluate.seed == 19
    assert train.num_bodies == evaluate.num_bodies == 1


def test_evaluation_uses_scenario_seed_and_exact_checkpoint(monkeypatch, tmp_path):
    events = []
    runner = SimpleNamespace(
        env=SimpleNamespace(close=lambda: events.append("close")), am=None,
        training_state_file_name="default-policy.ckpt",
        adaptation_module_file_name="default-adaptation.ckpt",
    )
    monkeypatch.setattr(benchmark, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(benchmark, "load_trained_modules", lambda *args: {
        "actor_model": {"weights": np.array([1.0])},
        "metadata": {"stage": "policy", "experiment": "ppo_nominal",
                     "policy_epoch": 50, "config_id": "config-1",
                     "resolved_config": {"seed": 7, "rl": {}}},
    })

    def build(experiment, seed, **options):
        assert experiment == "ppo_nominal" and seed == 7
        assert options["view_env"] == 2 and options["viewer_port"] == 8090
        return runner

    def evaluate(actual, **options):
        assert actual is runner and options["seed"] == 19
        assert runner.training_state_file_name == "renamed.ckpt"
        assert runner.evaluation_checkpoint == tmp_path / "renamed.ckpt"
        assert runner.ckpt_dir == str(tmp_path)
        assert options["source"] == "zero"
        assert options["output"].name == "scenarios"
        assert "train7_policy050_" in options["output"].parent.name
        assert options["output"].parent.name.endswith("_eval19_zero_core")
        options["output"].mkdir(parents=True)
        (options["output"] / "scenario.ckpt").write_text("state")
        events.append("evaluate")
        return [{"seed": 19}]

    monkeypatch.setattr(benchmark, "build_runner", build)
    monkeypatch.setattr(benchmark, "evaluate_suite", evaluate)
    benchmark.main(["evaluate", "--checkpoint", str(tmp_path / "renamed.ckpt"),
                    "--seed", "19", "--view-env", "2", "--viewer-port", "8090"])
    assert events == ["evaluate", "close"]
    results = list((tmp_path / "results").glob("*/results.json"))
    assert len(results) == 1
    assert (results[0].parent / "scenarios/scenario.ckpt").is_file()


def test_failed_training_closes_environment_and_finishes_wandb(monkeypatch):
    events = []

    def fail(**kwargs):
        raise RuntimeError("training failed")

    runner = SimpleNamespace(
        env=SimpleNamespace(close=lambda: events.append("close")), learn=fail,
    )
    monkeypatch.setattr(benchmark, "build_runner", lambda *args, **kwargs: runner)
    monkeypatch.setattr(benchmark.wandb, "finish", lambda: events.append("finish"))
    with pytest.raises(RuntimeError, match="training failed"):
        benchmark.main(["train", "--experiment", "ppo_nominal", "--wandb"])
    assert events == ["close", "finish"]


def test_runner_options_and_default_architecture(monkeypatch):
    captured = {}
    env = SimpleNamespace(close=lambda: None)

    def build_env(**kwargs):
        captured.update(kwargs)
        return env

    monkeypatch.setattr(benchmark, "AstrobeeEnvVectorized", build_env)
    monkeypatch.setattr(benchmark, "OraclePlannerRL", lambda *args, **kwargs: "planner")
    monkeypatch.setattr(benchmark, "make_runner", lambda *args, **kwargs: kwargs["config"])
    config = benchmark.build_runner(
        "ppo_nominal", 7, log=False, viewer=True, view_env=2,
        overrides={"am_architecture": "transformer"},
    )
    assert config.am_architecture == "transformer"
    assert not config.use_adaptive_approach
    assert captured["args"].viewer and captured["args"].view_env == 2
    assert not captured["args"].log


def test_runner_setup_failure_closes_environment(monkeypatch):
    closed = []
    env = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(benchmark, "AstrobeeEnvVectorized", lambda **kwargs: env)

    def fail(*args, **kwargs):
        raise RuntimeError("setup failed")

    monkeypatch.setattr(benchmark, "OraclePlannerRL", fail)
    with pytest.raises(RuntimeError, match="setup failed"):
        benchmark.build_runner("ppo_nominal", 7)
    assert closed == [True]


def test_evaluation_directories_separate_checkpoints_configs_and_seeds():
    policy = {
        "metadata": {"experiment": "rma_cnn", "resolved_config": {"seed": 7},
                     "policy_epoch": 50, "config_id": "config-1"},
        "actor_model": {"weights": np.array([1.0, 2.0])},
    }
    adaptation = {
        "metadata": {"adaptation_epoch": 10, "config_id": "config-1"},
        "am_model": {"weights": np.array([3.0])},
    }

    def directory(policy=policy, adaptation=adaptation, **options):
        return benchmark.evaluation_directory(
            policy, adaptation, **{"source": "estimated", "suite": "core", "seed": 10000, **options},
        )

    original = directory()
    assert "policy050_adapt010_" in original.name
    assert directory(deepcopy(policy), deepcopy(adaptation)) == original
    assert directory(seed=20000) != original
    assert directory(suite="stress") != original
    assert directory(adaptation=None, source="privileged") != original
    for section, key, value in (
        ("metadata", "policy_epoch", 100),
        ("metadata", "config_id", "config-2"),
        ("actor_model", "weights", np.array([2.0, 3.0])),
    ):
        changed = deepcopy(policy)
        changed[section][key] = value
        assert directory(policy=changed) != original
    for section, key, value in (
        ("metadata", "adaptation_epoch", 20),
        ("am_model", "weights", np.array([4.0])),
    ):
        changed = deepcopy(adaptation)
        changed[section][key] = value
        assert directory(adaptation=changed) != original
