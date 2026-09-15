"""Validate raw run coverage and regenerate canonical data and publication artifacts."""

import argparse
import json
import math
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import t

from .common import (
    EXPERIMENTS, digest, evaluation_trial_count, git_info, job_id, jobs, load_config, write_json,
)


def records(path):
    return (
        [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if path.exists()
        else []
    )


def wilson(successes, n, z=1.959963984540054):
    if not n:
        return (None, None)
    p = successes / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - radius, center + radius


def validate_rows(config, job, rows):
    protocol = config["protocol"]
    experiment = protocol["experiment"]
    if experiment == "exp1_scaling":
        if len(rows) == 1 and rows[0].get("status") == "oom":
            return
        actual = {(r.get("repeat"), r.get("num_envs")) for r in rows if r.get("status") == "ok"}
        expected = {(i, job["batch_size"]) for i in range(protocol["repeats"])}
    else:
        conditions = (
            [job["condition"]]
            if "condition" in job
            else list(config["common"]["evaluation_distributions"])
        )
        if experiment == "exp4_docking":
            trial_key = (
                "default_trials" if job.get("condition") == "default" else "sensitivity_trials"
            )
            count = protocol[trial_key]
        else:
            count = protocol["trials"]
        expected = {
            (condition, config["common"]["evaluation_seed_start"] + i)
            for condition in conditions
            for i in range(
                count if experiment == "exp4_docking" else evaluation_trial_count(config, condition)
            )
        }
        actual = {(r.get("condition"), r.get("evaluation_seed")) for r in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError(
            f"{job_id(job)}: raw rows incomplete/duplicated; expected {len(expected)}, got {len(rows)}; missing {sorted(expected - actual)[:5]}"
        )
    if experiment != "exp1_scaling":
        required = {
            "success",
            "final_position_error",
            "final_attitude_error",
            "control_effort",
            "termination_reason",
        }
        for row in rows:
            if not required <= row.keys():
                raise ValueError(f"Missing trial fields: {required - row.keys()}")


def validate_run_records(directory, config, meta, rows):
    """Apply the same completed-run checks during aggregation and resume."""
    validate_rows(config, meta["job"], rows)
    if (
        config["protocol"]["experiment"] != "exp3_rl_robustness"
        or meta["job"].get("regime") == "baseline"
    ):
        return
    training = records(directory / "training.jsonl")
    algorithm = meta["job"]["controller"]
    settings = config["common"]["training"][algorithm.upper()]
    count = config["protocol"].get("training_num_envs", {}).get(
        algorithm, config["common"]["env"]["environment"]["num_envs"]
    )
    budget = settings.get(
        "total_transitions", settings.get("epochs", 0) * settings.get("steps_per_epoch", 0) * count
    )
    axis = [row["environment_steps"] for row in training]
    checkpoint = meta.get("checkpoint_path")
    if (
        not axis
        or axis[-1] != budget
        or any(b <= a for a, b in zip([0] + axis, axis))
        or not checkpoint
        or not (directory / checkpoint).exists()
    ):
        raise ValueError(
            f"{directory}: missing checkpoint, incomplete budget or invalid training step axis"
        )


def discover(root, mode, allow_partial=False):
    accepted = []
    errors = []
    seen = set()
    configs = {}
    ignored = []
    for path in sorted(Path(root).glob(f"{mode}/*/*/metadata.json")):
        meta = json.loads(path.read_text())
        if meta["status"] != "complete":
            ignored.append(str(path.parent))
            continue
        config = yaml.safe_load((path.parent / "resolved_config.yaml").read_text())
        exp = meta["experiment"]
        identity = (exp, meta["job_id"])
        if identity in seen:
            raise ValueError(
                f"Duplicate completed job {identity}; select a single campaign root, never silently choose a repeat"
            )
        if exp not in EXPERIMENTS:
            raise ValueError(f"Unknown experiment {exp}")
        if meta["config_id"] != digest(config):
            raise ValueError(f"Config digest mismatch: {path}")
        if mode == "paper" and digest(config) != digest(load_config(exp, mode="paper")):
            raise ValueError(f"{path}: does not match the checked-in paper configuration")
        if exp in configs and digest(configs[exp]) != digest(config):
            raise ValueError(f"Mixed configs for {exp}; use separate campaign roots")
        rows = records(path.parent / "metrics.jsonl")
        try:
            validate_run_records(path.parent, config, meta, rows)
        except ValueError as error:
            if not allow_partial:
                raise
            errors.append(str(error))
            continue
        configs[exp] = config
        seen.add(identity)
        accepted.append((path.parent, meta, rows))
    for exp in EXPERIMENTS:
        config = configs.get(exp, load_config(exp, mode=mode))
        for job in jobs(config):
            if (exp, job_id(job)) not in seen:
                errors.append(f"Missing {exp}/{job_id(job)}")
    if errors and not allow_partial:
        raise ValueError(
            "\n".join(errors) + "\nUse --allow-partial only for explicit incomplete artifacts."
        )
    return accepted, errors, ignored


def summarize(frame, groups):
    numeric = [
        c
        for c in (
            "mean_position_error",
            "final_position_error",
            "mean_attitude_error",
            "final_attitude_error",
            "control_effort",
            "completion_time",
            "peak_contact_force",
            "settling_time",
            "contact_impulse",
            "cumulative_contact_duration",
        )
        if c in frame
    ]
    rows = []
    for identity, subset in frame.groupby(groups, dropna=False, sort=True):
        identity = identity if isinstance(identity, tuple) else (identity,)
        n = len(subset)
        successes = int(subset.success.sum())
        low, high = wilson(successes, n)
        row = dict(zip(groups, identity))
        row.update(n=n, success_rate=successes / n, success_ci_low=low, success_ci_high=high)
        if "training_seed" in subset and subset.training_seed.notna().any():
            rates = subset.groupby("training_seed").success.mean()
            row["n_training_seeds"] = len(rates)
            row["success_seed_mean"] = float(rates.mean())
            row["success_seed_std"] = float(rates.std()) if len(rates) > 1 else None
            half = (
                float(t.ppf(0.975, len(rates) - 1) * rates.std() / np.sqrt(len(rates)))
                if len(rates) > 1
                else None
            )
            row["success_seed_ci_low"] = max(0.0, rates.mean() - half) if half is not None else None
            row["success_seed_ci_high"] = (
                min(1.0, rates.mean() + half) if half is not None else None
            )
        for metric in numeric:
            values = pd.to_numeric(subset[metric], errors="coerce").dropna()
            row.update(
                {
                    metric + "_n": len(values),
                    metric + "_mean": values.mean(),
                    metric + "_std": values.std(),
                    metric + "_median": values.median(),
                    metric + "_q25": values.quantile(0.25),
                    metric + "_q75": values.quantile(0.75),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def learning_curves(frame):
    rows = []
    for identity, group in frame.groupby(["controller", "regime"]):
        curves = [g.sort_values("environment_steps") for _, g in group.groupby("training_seed")]
        low = max(g.environment_steps.min() for g in curves)
        high = min(g.environment_steps.max() for g in curves)
        grid = np.unique(np.concatenate([g.environment_steps.to_numpy() for g in curves]))
        grid = grid[(grid >= low) & (grid <= high)]
        for metric in ("mean_episodic_returns", "success_rate", "mean_lateral_error",
                       "mean_angle_error", "critic_loss_mean", "actor_loss_mean",
                       "true_kl_mean", "clip_fraction", "explained_variance", "alpha", "entropy"):
            if metric not in group or not group[metric].notna().any():
                continue
            values = np.stack([np.interp(grid, g.environment_steps, g[metric]) for g in curves])
            for index, step in enumerate(grid):
                rows.append(
                    dict(
                        controller=identity[0],
                        regime=identity[1],
                        environment_steps=int(step),
                        metric=metric,
                        mean=values[:, index].mean(),
                        std=values[:, index].std(ddof=1) if len(curves) > 1 else None,
                        n_seeds=len(curves),
                    )
                )
    return pd.DataFrame(rows)


def training_health(records):
    """Report numerical health and episode-weighted convergence, not a quality claim."""
    grouped = defaultdict(list)
    for row in records:
        grouped[row["run_id"]].append(row)
    output = []
    fields = ("epoch_seconds", "mean_reward", "mean_lateral_error", "mean_angle_error",
              "mean_episodic_returns", "success_rate", "actor_loss_mean", "critic_loss_mean",
              "true_kl_mean", "explained_variance", "clip_fraction", "alpha", "entropy")
    for run_id, rows in grouped.items():
        rows.sort(key=lambda row: row["environment_steps"])
        first = rows[0]
        width = max(1, len(rows) // 10)
        row = dict(run_id=run_id, controller=first["controller"], regime=first["regime"],
                   training_seed=first["training_seed"], updates=len(rows),
                   environment_steps=rows[-1]["environment_steps"])
        invalid = sum(1 for update in rows for field in fields if field in update
                      and (update[field] is None or not np.isfinite(update[field])))
        row.update(nonfinite_diagnostics=invalid, numerical_health_ok=invalid == 0)
        for label, window in (("early", rows[:width]), ("late", rows[-width:])):
            completed = sum(update.get("completed_episode_count", 0) for update in window)
            successes = sum(update.get("success_termination_step_count", 0) for update in window)
            row[label + "_completed_episodes"] = completed
            row[label + "_success_rate"] = successes / completed if completed else None
            for metric in ("mean_lateral_error", "mean_angle_error", "mean_episodic_returns"):
                values = [update[metric] for update in window
                          if update.get(metric) is not None and np.isfinite(update[metric])]
                row[label + "_" + metric] = float(np.mean(values)) if values else None
        output.append(row)
    return pd.DataFrame(output)


def _aggregate(root, output, *, mode="paper", allow_partial=False):
    runs, missing, ignored = discover(root, mode, allow_partial)
    output = Path(output)
    data = output / "data"
    data.mkdir(parents=True, exist_ok=True)
    frames = defaultdict(list)
    training = []
    inputs = defaultdict(list)
    representatives = []
    for path, meta, rows in runs:
        exp = meta["experiment"]
        inputs[exp].append(str(path.resolve()))
        for row in rows:
            frames[exp].append({**row, "run_id": meta["run_id"]})
        for row in records(path / "training.jsonl"):
            training.append({**row, "training_seed": meta["seed"], "run_id": meta["run_id"]})
        if exp == "exp4_docking" and meta["job"]["condition"] == "default":
            executed = [
                r
                for r in rows
                if r.get("peak_contact_force") is not None and r.get("episode_length", 1) > 0
            ]
            successes = [r for r in executed if r["success"]]
            candidates = successes or executed
            if not candidates:
                continue  # Initial-penetration failures have no physical trajectory to plot.
            # Lower median by peak force, tie-broken by evaluation seed; explicit failure fallback.
            chosen = sorted(
                candidates, key=lambda r: (r["peak_contact_force"], r["evaluation_seed"])
            )[(len(candidates) - 1) // 2]
            trace = json.loads((path / "traces" / f"{chosen['trial']:04d}.json").read_text())
            representatives.extend(
                {
                    **r,
                    "evaluation_seed": chosen["evaluation_seed"],
                    "run_id": meta["run_id"],
                    "selection": "median_success_peak_force"
                    if successes
                    else "median_failure_peak_force",
                }
                for r in trace
            )
    exported = {}

    def save(name, frame, exp):
        frame.to_csv(data / f"{name}.csv", index=False, float_format="%.17g")
        exported[name] = exp
        return frame

    for exp, rows in frames.items():
        frame = pd.DataFrame(rows)
        if exp == "exp1_scaling":
            good = frame[frame.status == "ok"].copy()
            if not good.empty:
                base = good.loc[good.num_envs == 1, "env_steps_per_second"].mean()
                good["speedup"] = good.env_steps_per_second / base
                good["parallel_efficiency"] = good.speedup / good.num_envs
                frame = pd.concat([good, frame[frame.status != "ok"]], ignore_index=True)
            save("exp1_scaling", frame, exp)
        else:
            prefix = {
                "exp2_fault_robustness": "exp2",
                "exp3_rl_robustness": "exp3",
                "exp4_docking": "exp4",
                "spacecraft_portability": "spacecraft_portability",
            }[exp]
            if prefix == "exp3":
                trial_name = "exp3_evaluation_trials"
            elif prefix == "spacecraft_portability":
                trial_name = prefix
            else:
                trial_name = f"{prefix}_trials"
            save(trial_name, frame, exp)
            groups = [c for c in ("spacecraft", "controller", "regime", "condition") if c in frame]
            summary = save(prefix + "_summary", summarize(frame, groups), exp)
            write_json(
                output / "paper/summaries" / f"{prefix}_summary.json", summary.to_dict("records")
            )
            if prefix == "exp4":
                save("exp4_contact_sensitivity", summary, exp)
    if training:
        frame = save("exp3_training_curves", pd.DataFrame(training), "exp3_rl_robustness")
        save("exp3_learning_summary", learning_curves(frame), "exp3_rl_robustness")
        save("exp3_training_health", training_health(training), "exp3_rl_robustness")
    if representatives:
        save("exp4_representative", pd.DataFrame(representatives), "exp4_docking")
    if "exp1_scaling" in frames:
        write_json(output / "paper/summaries/exp1_summary.json", frames["exp1_scaling"])
    manifest = dict(
        mode=mode,
        generated_at=datetime.now(timezone.utc).isoformat(),
        partial=bool(missing),
        missing=missing,
        ignored_incomplete_runs=ignored,
        artifacts=[
            dict(artifact=f"data/{name}.csv", experiment=exp, input_runs=inputs[exp])
            for name, exp in exported.items()
        ],
        **git_info(),
    )
    from .make_figures import make_figures
    from .make_tables import make_tables

    for artifact, exp in make_figures(
        data,
        output / "paper/figures",
        label=(mode.upper() + (" — INCOMPLETE DATA" if missing else ""))
        if mode != "paper" or missing
        else None,
    ) + make_tables(data, output / "paper/tables"):
        if artifact.suffix == ".tex":
            artifact.write_text(
                f"% Data mode: {mode}; partial: {bool(missing)}\n" + artifact.read_text()
            )
        manifest["artifacts"].append(
            dict(artifact=str(artifact.relative_to(output)), experiment=exp, input_runs=inputs[exp])
        )
    for source in sorted((output / "paper/summaries").glob("*.json")):
        prefix = source.stem.removesuffix("_summary")
        exp = next((e for e in EXPERIMENTS if e.startswith(prefix)), None)
        if exp:
            manifest["artifacts"].append(
                dict(
                    artifact=str(source.relative_to(output)), experiment=exp, input_runs=inputs[exp]
                )
            )
    write_json(output / "manifest.json", manifest)
    return manifest


def aggregate(root, output, *, mode="paper", allow_partial=False):
    """Generate in a clean staging directory, then publish only this artifact set."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="demonstration-use-cases-artifacts-", dir=output.parent
    ) as staging:
        staging = Path(staging)
        manifest = _aggregate(root, staging, mode=mode, allow_partial=allow_partial)
        previous = output / "manifest.json"
        if previous.exists():
            old = json.loads(previous.read_text())
            current = {a["artifact"] for a in manifest["artifacts"]}
            for artifact in old.get("artifacts", []):
                name = artifact["artifact"]
                path = output / name
                if name not in current and path.resolve().is_relative_to(output.resolve()):
                    path.unlink(missing_ok=True)
        for path in staging.rglob("*"):
            if path.is_file() and path.name != "manifest.json":
                target = output / path.relative_to(staging)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        write_json(output / "manifest.json", manifest)
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path("results/demonstration_use_cases"))
    p.add_argument("--output", type=Path, default=Path("artifacts/demonstration_use_cases"))
    p.add_argument("--mode", choices=["paper", "smoke", "development"], default="paper")
    p.add_argument("--allow-partial", action="store_true")
    args = p.parse_args(argv)
    result = aggregate(args.results, args.output, mode=args.mode, allow_partial=args.allow_partial)
    print(f"Wrote {len(result['artifacts'])} artifacts; partial={result['partial']}")


if __name__ == "__main__":
    main()
