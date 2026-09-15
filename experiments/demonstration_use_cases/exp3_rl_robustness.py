"""Train every algorithm/regime/seed, then evaluate paired held-out episodes."""

import json
import subprocess
import sys
from time import perf_counter

import numpy as np
import wandb

from .common import Run, execute, parser, write_json
from .runtime import build_runner, evaluate


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
        run.update(checkpoint_path=str(checkpoint.relative_to(run.path)))
        # The trained actor is reused directly, avoiding checkpoint/config identity drift.
        evaluate(config, job, run, actor=runner.agent.actor)
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


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--worker":
        run = Run.open_existing(argv[1])
        train_evaluate(run.config, run.job, run)
        return
    execute("exp3_rl_robustness", parser("exp3_rl_robustness").parse_args(argv), isolated_training)


if __name__ == "__main__":
    main()
