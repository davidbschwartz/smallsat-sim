"""Evaluation-only CLI restores archived policies without touching their training runs."""
import hashlib
import json

import pytest
import yaml

from experiments.paper_benchmarks.common import load_config, run_directory, write_json
from experiments.paper_benchmarks.exp3_rl_robustness import main, train_evaluate
from experiments.paper_benchmarks.aggregate import records, validate_run_records
from smallsat_sim.controllers.rl.runners.on_policy_runner import OnPolicyRunner
from smallsat_sim.controllers.rl.runners.off_policy_runner import OffPolicyRunner


def fingerprints(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("algorithm", ["ppo", "sac"])
def test_evaluate_archived_run_without_training(tmp_path, monkeypatch, algorithm):
    config = load_config("exp3_rl_robustness", mode="smoke")
    job = dict(controller=algorithm, regime="nominal", seed=0, spacecraft="astrobee")
    with run_directory(tmp_path / "training", config, job) as run:
        write_json(run.path / "job.json", job)
        train_evaluate(config, job, run)
    # Moving the directory must not break strict checkpoint identity validation.
    archived = tmp_path / "archived_run"
    run.path.rename(archived)
    before = fingerprints(archived)

    def no_training(*args, **kwargs):
        raise AssertionError("Evaluation-only must not train")

    monkeypatch.setattr(OnPolicyRunner, "learn", no_training)
    monkeypatch.setattr(OffPolicyRunner, "learn", no_training)
    config["common"]["evaluation_seed_start"] = 50000
    config["common"]["episode_steps"] = 4
    config["protocol"]["evaluation_trials"]["combined"] = 1
    requested = tmp_path / "evaluation.yaml"
    requested.write_text(yaml.safe_dump(config))
    output = tmp_path / "evaluation"
    main(["--evaluate-run", str(archived), "--output", str(output), "--config", str(requested)])
    metadata_path, = output.rglob("metadata.json")
    meta = json.loads(metadata_path.read_text())
    assert meta["status"] == "complete" and meta["evaluation_only"]
    assert meta["source_run"] == str(archived)
    cfg = yaml.safe_load((metadata_path.parent / "resolved_config.yaml").read_text())
    rows = records(metadata_path.parent / "metrics.jsonl")
    assert len(rows) == 8
    assert {r["evaluation_seed"] for r in rows} == {50000, 50001}
    assert all(r["episode_length"] <= 4 for r in rows)
    validate_run_records(metadata_path.parent, cfg, meta, rows)
    assert fingerprints(archived) == before
    config["common"]["training"]["PPO"]["epochs"] += 1
    requested.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="only scenario"):
        main(["--evaluate-run", str(archived), "--output", str(output), "--config", str(requested)])
