"""Train every algorithm/regime/seed, then evaluate paired held-out episodes."""

from copy import deepcopy
from pathlib import Path
import shutil

import json
import subprocess
import sys
from time import perf_counter

import numpy as np
import wandb
import yaml

from .common import Run, execute, parser, write_json, run_directory, validate_config
from .runtime import build_runner, evaluate
from .mjx_evaluation import evaluate_policy


def train_evaluate(config, job, run):
    if job["regime"] == "baseline":
        return evaluate(config, job, run)
    runner = build_runner(config, job, run)
    try:
        training_start = perf_counter()
        runner.learn(mode="fresh")
        run.update(training_wall_seconds=perf_counter() - training_start)
        path = run.path / "metrics" / f"{runner.env.run_name}_seed{job['seed']}.jsonl"
        elapsed = 0.0
        for line in path.read_text().splitlines():
            row = json.loads(line)
            elapsed += row["epoch_seconds"]
            row["environment_steps"] = int(
                row.get(
                    "environment_transitions",
                    row["epoch"]
                    * runner.env.num_envs
                    * config["common"]["training"][job["controller"].upper()].get(
                        "steps_per_epoch", 0
                    ),
                )
            )
            row["wall_seconds"] = elapsed
            row.update({k: job[k] for k in ("controller", "regime", "seed", "spacecraft")})
            run.append(row, "training.jsonl")
        checkpoint = run.path / "checkpoint" / runner.training_state_file_name
        policy_checkpoint = run.path / "checkpoint" / getattr(runner, "policy_file_name", runner.training_state_file_name)
        if not policy_checkpoint.exists():
            raise RuntimeError(f"Final policy checkpoint was not saved: {policy_checkpoint}")
        run.update(checkpoint_path=str(checkpoint.relative_to(run.path)),
                   policy_checkpoint_path=str(policy_checkpoint.relative_to(run.path)))
        evaluate_policy(config, job, run, runner)
        if runner.env.use_wandb:
            rows = [json.loads(line) for line in (run.path / "metrics.jsonl").read_text().splitlines()]
            for condition in config["common"]["evaluation_distributions"]:
                trials = [row for row in rows if row["condition"] == condition]
                wandb.log({f"evaluation/{condition}/{metric}": float(np.mean([row[metric] for row in trials]))
                           for metric in ("success", "final_position_error", "final_attitude_error", "control_effort")})
    finally:
        runner.env.close()
        if runner.env.use_wandb:
            wandb.finish()


def isolated_training(config, job, run):
    """Release compiled programs and accelerator allocations after each policy."""
    write_json(run.path / "job.json", job)
    with (run.path / "worker.log").open("w") as log:
        result = subprocess.run(
            [sys.executable, "-u", "-m", "experiments.demonstration_use_cases.exp3_rl_robustness",
             "--worker", str(run.path.resolve())], stdout=log, stderr=subprocess.STDOUT,
        )
    # The worker records checkpoint/timing fields; retain them when the parent
    # context manager commits the final run status.
    run.meta = json.loads((run.path / "metadata.json").read_text())
    if result.returncode:
        raise RuntimeError(f"RL worker exited with {result.returncode}; see {run.path / 'worker.log'}")



def evaluate_saved_run(directory, output, config_path=None):
    """Evaluate an archived policy into a fresh run; never train or alter its source."""
    source = Run.open_existing(Path(directory).resolve())
    if (source.config["protocol"]["experiment"] != "exp3_rl_robustness"
            or source.job.get("regime") == "baseline"):
        raise ValueError("--evaluate-run requires a saved RL robustness policy run")
    checkpoint_name = source.meta.get("policy_checkpoint_path") or source.meta.get("checkpoint_path")
    if not checkpoint_name or not (source.path / checkpoint_name).exists():
        raise ValueError("Saved run has no available policy checkpoint")
    checkpoint = (source.path / checkpoint_name).resolve()
    config = deepcopy(source.config)
    if config_path is not None:
        requested = yaml.safe_load(Path(config_path).read_text())
        # Evaluation changes must not silently change the trained task or policy.
        for section, keys in {
            "common": ("evaluation_distributions", "evaluation_seed_start", "episode_steps"),
            "protocol": ("trials", "evaluation_trials", "evaluation_batch_size", "evaluation_backend"),
        }.items():
            for key in keys:
                if key in requested[section]:
                    config[section][key] = requested[section][key]
                else:
                    config[section].pop(key, None)
        config["mode"] = requested.get("mode", source.config["mode"])
        if config != requested:
            raise ValueError("Evaluation config may change only scenario distributions, seeds, counts, batch size, and horizon")
        config["mode"] = "development"
    validate_config(config)
    with run_directory(output, config, source.job) as run:
        write_json(run.path / "job.json", source.job)
        run.update(evaluation_only=True, source_run=str(source.path),
                   source_config_id=source.meta["config_id"],
                   checkpoint_path=str(checkpoint), policy_checkpoint_path=str(checkpoint))
        if (source.path / "training.jsonl").exists():
            shutil.copyfile(source.path / "training.jsonl", run.path / "training.jsonl")
        runner = build_runner(source.config, source.job, run, saved_run=source)
        try:
            runner.evaluation_checkpoint = checkpoint
            evaluate_policy(config, source.job, run, runner)
        finally:
            runner.env.close()
    print(f"Evaluation-only results: {run.path}", flush=True)
    return run.path

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--worker":
        run = Run.open_existing(argv[1])
        train_evaluate(run.config, run.job, run)
        return
    cli = parser("exp3_rl_robustness")
    cli.add_argument("--evaluate-run", type=Path, help="Evaluate a saved run without training; write fresh results under --output")
    args = cli.parse_args(argv)
    if args.evaluate_run is not None:
        if args.smoke or args.paper or args.job_id or args.resume or args.plan:
            cli.error("--evaluate-run uses the saved run; do not combine it with mode, job, resume, or plan options")
        evaluate_saved_run(args.evaluate_run, args.output, args.config)
    else:
        execute("exp3_rl_robustness", args, isolated_training)


if __name__ == "__main__":
    main()
