"""Train every algorithm/regime/seed, then evaluate paired held-out episodes."""

import json

from .common import execute, parser
from .runtime import build_runner, evaluate


def train_evaluate(config, job, run):
    if job["regime"] == "baseline":
        return evaluate(config, job, run)
    runner = build_runner(config, job, run)
    try:
        runner.learn(mode="fresh")
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
    finally:
        runner.env.close()


def main(argv=None):
    execute("exp3_rl_robustness", parser("exp3_rl_robustness").parse_args(argv), train_evaluate)


if __name__ == "__main__":
    main()
