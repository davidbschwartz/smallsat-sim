"""Train, evaluate or deploy a named RL benchmark using python -m."""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json

import jax
import numpy as np
import wandb

from .evaluation import evaluate_suite
from experiments.astrobee_RL import run
from smallsat_sim.configuration import read_yaml, settings
from smallsat_sim.controllers.rl.runners.factory import make_runner
from smallsat_sim.controllers.rl.runners.runner_utils import (
    checkpoint_exists, load_trained_modules, resolve_checkpoint_paths,
)
from smallsat_sim.envs.vehicles.astrobee_rl.env import AstrobeeEnvVectorized
from smallsat_sim.envs.vehicles.astrobee_rl.config import resolve_config
from smallsat_sim.envs.rendering.rollout import add_visualization_args
from smallsat_sim.planners.oracle.oracle_rl import OraclePlannerRL

PRESETS = read_yaml("benchmarks.yaml")["experiments"]
RESULTS_DIR = Path("experiments/rl_results")


def build_runner(
    experiment, seed, *, overrides=None, log=True, wandb=False,
    viewer=False, video=False, view_env=0, viewer_port=8080,
    video_duration=10.0, video_dir="videos",
):
    preset = settings(PRESETS[experiment])
    args = SimpleNamespace(
        headless=True, num_bodies=1, log=log, wandb=wandb,
        viewer=viewer, video=video, view_env=view_env, viewer_port=viewer_port,
        video_duration=video_duration, video_dir=video_dir,
    )
    config = resolve_config(
        seed=seed,
        algorithm=preset.algorithm,
        train_with_failures=preset.randomized,
        use_pretrained=False,
        use_adaptive_approach=preset.adaptation is not None,
        am_architecture=preset.adaptation,
        overrides=overrides,
    )
    env = AstrobeeEnvVectorized(args=args, run_name=experiment, config=config.env)
    try:
        planner = OraclePlannerRL(env, radius=0.0)
        return make_runner(env, planner, config=config.training)
    except BaseException:
        env.close()
        raise


@contextmanager
def benchmark_runner(args, experiment, seed, overrides):
    runner = build_runner(
        experiment, seed, overrides=overrides, log=args.log, wandb=args.wandb,
        viewer=args.viewer, video=args.video, view_env=args.view_env,
        viewer_port=args.viewer_port, video_duration=args.video_duration,
        video_dir=args.video_dir,
    )
    try:
        yield runner
        if hasattr(runner.env, "logger"):
            runner.env.logger.save_log()
    finally:
        runner.env.close()


def train(args):
    if args.teacher is not None and (args.stage != "adaptation" or args.mode != "fresh"):
        args.parser.error("--teacher requires --stage adaptation --mode fresh")
    if args.stage == "adaptation" and PRESETS[args.experiment]["adaptation"] is None:
        args.parser.error("The adaptation stage requires an adaptive experiment")
    overrides = read_yaml(args.config) if args.config else {}
    with benchmark_runner(args, args.experiment, args.seed, overrides) as runner:
        if args.teacher is not None:
            runner.initialize_from_teacher(args.teacher)
        if args.stage in ("policy", "all"):
            runner.learn(mode=args.mode)
        if runner.am is not None and args.stage in ("adaptation", "all"):
            # A missing adaptation checkpoint starts fresh even after policy resume.
            mode = args.mode if checkpoint_exists(
                runner.ckpt_dir, runner.adaptation_module_file_name,
            ) else "fresh"
            runner.train_adaptation_module_on_policy(mode=mode)


def evaluation_directory(policy, adaptation, *, source, suite, seed):
    """Identify evaluated weights, configuration and seeds independently of filenames."""
    metadata = policy["metadata"]
    progress = f"policy{int(metadata['policy_epoch']):03d}"
    weights = {"policy": policy["actor_model"]}
    digest = hashlib.sha256(metadata["config_id"].encode())
    if adaptation is not None:
        progress += f"_adapt{int(adaptation['metadata']['adaptation_epoch']):03d}"
        weights["adaptation"] = adaptation["am_model"]
        digest.update(adaptation["metadata"]["config_id"].encode())
    for path, value in jax.tree_util.tree_flatten_with_path(weights)[0]:
        array = np.asarray(value)
        digest.update(str(path).encode())
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    name = (
        f"{metadata['experiment']}_train{metadata['resolved_config']['seed']}_"
        f"{progress}_{digest.hexdigest()[:12]}_eval{seed}_{source}_{suite}"
    )
    return RESULTS_DIR / name


def evaluate(args):
    payload = load_trained_modules(args.checkpoint.parent, args.checkpoint.name)
    metadata = payload["metadata"]
    config = metadata["resolved_config"]
    experiment, training_seed = metadata["experiment"], config["seed"]
    with benchmark_runner(args, experiment, training_seed, config["rl"]) as runner:
        policy, adaptation = resolve_checkpoint_paths(
            args.checkpoint, stage=metadata["stage"],
            policy_filename=runner.training_state_file_name,
            adaptation_filename=runner.adaptation_module_file_name,
        )
        runner.ckpt_dir = str(args.checkpoint.parent)
        runner.training_state_file_name = policy.name
        runner.evaluation_checkpoint = policy
        runner.adaptation_module_file_name = adaptation.name
        source = args.context_source if runner.am is not None else "zero"
        policy_payload = payload if policy == args.checkpoint else load_trained_modules(
            policy.parent, policy.name,
        )
        adaptation_payload = None
        if source == "estimated":
            adaptation_payload = (
                payload if adaptation == args.checkpoint
                else load_trained_modules(adaptation.parent, adaptation.name)
            )
        output = evaluation_directory(
            policy_payload, adaptation_payload,
            source=source, suite=args.suite, seed=args.seed,
        )
        rows = evaluate_suite(
            runner, source=source, suite=args.suite, seed=args.seed,
            output=output / "scenarios",
        )
        output.mkdir(parents=True, exist_ok=True)
        (output / "results.json").write_text(json.dumps(rows, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    training = commands.add_parser("train", help="Train a named experiment")
    training.add_argument("--experiment", choices=tuple(PRESETS), required=True)
    training.add_argument("--seed", type=int, default=42, help="Training seed")
    training.add_argument("--config", type=Path, help="YAML mapping of RL overrides")
    training.add_argument("--teacher", type=Path, help="Policy checkpoint for fresh adaptation training")
    training.add_argument("--mode", choices=("fresh", "resume"), default="fresh")
    training.add_argument("--stage", choices=("policy", "adaptation", "all"), default="all")
    training.set_defaults(handler=train, parser=training)

    evaluation = commands.add_parser("evaluate", help="Evaluate a saved policy")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--seed", type=int, default=10000, help="Evaluation scenario seed")
    evaluation.add_argument(
        "--context-source", choices=("estimated", "privileged", "zero"), default="estimated",
        help="Use history-based estimates, simulator residuals, or zero context",
    )
    evaluation.add_argument(
        "--suite", choices=("core", "stress"), default="core",
        help="core: individual faults; stress: core plus compound faults",
    )
    evaluation.set_defaults(handler=evaluate)

    deployment = commands.add_parser("deploy", help="Track a circular path with a saved policy")
    deployment.add_argument("--checkpoint", type=Path, required=True)
    deployment.set_defaults(handler=run)
    for command in (training, evaluation, deployment):
        command.set_defaults(headless=True, num_bodies=1)
        command.add_argument("--headless", action="store_true", default=True,
                             help="Run headless; --viewer enables browser visualization")
        command.add_argument("--wandb", action="store_true")
        command.add_argument("--log", action=argparse.BooleanOptionalAction, default=True)
        add_visualization_args(command)
        command.add_argument("--video", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.handler(args)
    finally:
        if args.wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
