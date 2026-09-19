"""Scientific invariants: observed-step aggregation, deterministic selection, raw coverage."""

import json
import hashlib
import sys

import numpy as np
import pandas as pd
import pytest

from .generate import ROOT, exact_learning, representative


def test_archived_campaign_exports_both_distribution_layouts(tmp_path, monkeypatch):
    from PIL import Image
    from . import generate

    source = ROOT / "smallsat-demonstration"
    if not (source / "paper-results/raw_runs").is_dir():
        pytest.skip("Requires the archived campaign source bundle")
    monkeypatch.setattr(sys, "argv", [
        "generate", "--source", str(source), "--output", str(tmp_path),
    ])
    generate.main()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    artifacts = {row["filename"]: row for row in manifest["artifacts"]}
    base = "figures/model_based_robustness/distributions"
    for suffix in ("pdf", "png"):
        horizontal, vertical = f"{base}.{suffix}", f"{base}_vertical.{suffix}"
        for name in (horizontal, vertical):
            assert (tmp_path / name).stat().st_size > 0
        assert artifacts[horizontal]["sources"] == artifacts[vertical]["sources"]
        assert artifacts[horizontal]["samples"] == artifacts[vertical]["samples"]
    with Image.open(tmp_path / f"{base}_vertical.png") as image:
        assert image.height > image.width


def test_rl_export_supports_source_outside_checkout(tmp_path, monkeypatch):
    from . import generate_rl
    from .generate import METHODS

    source = tmp_path / "external data"
    output = tmp_path / "publication output"
    data = source / "data"
    data.mkdir(parents=True)
    assert not source.resolve().is_relative_to(ROOT)
    training, trials = [], []
    conditions = ["nominal", "randomized_initial", "thrust_variation", "disturbance", "combined"]
    for controller, regime in METHODS:
        for seed in range(5):
            identity = dict(controller=controller, regime=regime, training_seed=seed)
            for step in [2097152, 4194304]:
                training.append(dict(identity, environment_steps=step, mean_episodic_returns=seed))
            for condition in conditions:
                trials.append(dict(identity, condition=condition, trial=0, success=True,
                                   run_id=f"{controller}-{regime}-{seed}"))
    training_path = data / "exp3_training_curves.csv"
    trials_path = data / "exp3_evaluation_trials.csv"
    pd.DataFrame(training).to_csv(training_path, index=False)
    pd.DataFrame(trials).to_csv(trials_path, index=False)
    monkeypatch.setattr(sys, "argv", ["generate_rl", "--source", str(source), "--output", str(output)])

    generate_rl.main()

    manifest = json.loads((output / "rl_v3_manifest.json").read_text())
    hashes = {(ROOT / name).resolve(): value for name, value in manifest["input_sha256"].items()}
    for path in [training_path, trials_path]:
        assert hashes[path.resolve()] == hashlib.sha256(path.read_bytes()).hexdigest()
    for artifact in manifest["artifacts"]:
        assert (output / artifact["filename"]).is_file()
        assert set(artifact["sources"]) <= set(manifest["input_sha256"])


def test_learning_never_interpolates_or_hides_seed_counts():
    f = pd.DataFrame(
        dict(
            controller=["ppo"] * 4,
            regime=["nominal"] * 4,
            training_seed=[0, 0, 1, 1],
            environment_steps=[10, 20, 10, 30],
            mean_episodic_returns=[2.0, 4.0, 6.0, 8.0],
        )
    )
    result = exact_learning(f).set_index("environment_steps")
    assert result.index.tolist() == [10, 20, 30]
    assert result.loc[10, "mean"] == 4
    assert result.n_seeds.tolist() == [2, 1, 1]
    assert np.isnan(result.loc[20, "std"])
    with pytest.raises(ValueError, match="Duplicate"):
        exact_learning(pd.concat([f, f.iloc[:1]]))


def test_representative_uses_successes_lower_median_and_seed_ties():
    f = pd.DataFrame(
        dict(
            success=[True, True, True, True, False],
            peak_contact_force=[3.0, 1.0, 2.0, 2.0, 0.0],
            evaluation_seed=[1, 2, 4, 3, 5],
            run_id=["a"] * 5,
        )
    )
    selected, rule = representative(f.sample(frac=1, random_state=4), "peak_contact_force")
    assert selected.evaluation_seed == 3
    assert rule == "lower median successful"
    f.success = False
    _, rule = representative(f, "peak_contact_force")
    assert "failed" in rule


def test_exported_campaign_matches_raw_trials():
    output = ROOT / "paper"
    if not (output / "manifest.json").exists():
        pytest.skip("Run generate.py first")
    audit = pd.read_csv(output / "tables/run_audit.csv")
    assert len(audit) == 43
    assert set(audit.status) == {"complete"}
    rl = pd.read_csv(output / "tables/exp3_rl_robustness_trials.csv")
    for (method, regime), g in rl[rl.regime != "baseline"].groupby(["controller", "regime"]):
        assert set(g.training_seed) == set(range(5))
        assert (g.groupby(["training_seed", "condition"]).size() == 100).all()
    manifest = json.loads((output / "manifest.json").read_text())
    for exp in [
        "exp2_fault_robustness",
        "exp3_rl_robustness",
        "exp4_docking",
        "spacecraft_portability",
    ]:
        artifact = next(a for a in manifest["artifacts"] if a["filename"] == f"tables/{exp}_trials.csv")
        raw_paths = [ROOT / p for p in artifact["sources"] if p.endswith("/metrics.jsonl")]
        assert raw_paths
        rows = [
            json.loads(line)
            for p in raw_paths
            for line in p.read_text().splitlines()
            if line.strip()
        ]
        trials = pd.read_csv(output / "tables" / f"{exp}_trials.csv")
        summary = pd.read_csv(output / "tables" / f"{exp}_summary.csv")
        assert len(trials) == len(rows) == summary.n.sum()
        assert (
            trials.success.sum() == sum(r["success"] for r in rows) == summary.success_count.sum()
        )
    scaling = pd.read_csv(output / "tables/scaling_measurements.csv")
    np.testing.assert_allclose(
        scaling.aggregate_sim_s_per_wall_s,
        scaling.env_steps_per_second * scaling.sim_dt * scaling.control_decimation,
    )
    np.testing.assert_allclose(
        scaling.per_environment_sim_s_per_wall_s,
        scaling.aggregate_sim_s_per_wall_s / scaling.num_envs,
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert not manifest["issues"]
    assert all((output / a["filename"]).is_file() for a in manifest["artifacts"])


def test_learning_windows_include_every_update_before_seed_aggregation():
    from .generate import windowed_learning

    frame = pd.DataFrame(
        dict(
            controller=["sac"] * 5,
            regime=["nominal"] * 5,
            training_seed=[0, 0, 0, 1, 1],
            environment_steps=[1, 2, 3, 2, 3],
            mean_episodic_returns=[2.0, 4.0, 8.0, 10.0, 12.0],
        )
    )
    per_seed, summary = windowed_learning(frame, 2)
    assert per_seed.recorded_updates.sum() == len(frame)
    first = summary.set_index("window_end").loc[2]
    assert first["mean"] == 6.5  # Equal seed weighting: mean([mean([2,4]),10]).
    assert first.n_seeds == 2
    assert summary.window_end.tolist() == [2, 4]
