"""Run or resume an RL-controlled Astrobee rollout."""

import argparse
from pathlib import Path

import jax.numpy as jnp

from smallsat_sim.controllers.rl.controller import RLController
from smallsat_sim.controllers.rl.runners.runner_utils import load_trained_modules
from smallsat_sim.envs.vehicles.astrobee_rl.env import AstrobeeEnvVectorized
from smallsat_sim.envs.vehicles.astrobee_rl.config import resolve_config
from smallsat_sim.planners.oracle.oracle_rl import OraclePlannerRL
from smallsat_sim.utils.helpers import get_args


def run(args):
    checkpoint = args.checkpoint
    if checkpoint is None:
        run_name, seed, overrides = "default", None, None
    else:
        payload = load_trained_modules(checkpoint.parent, checkpoint.name)
        metadata = payload["metadata"]
        run_name = metadata["experiment"]
        seed = metadata["resolved_config"]["seed"]
        overrides = metadata["resolved_config"]["rl"]

    config = resolve_config(
        seed=seed,
        overrides=overrides,
        init_pos=jnp.array([5.0, 0.0, 10.17]),
        max_start_offset=0.5,
    )
    env = AstrobeeEnvVectorized(args=args, run_name=run_name, config=config.env)
    try:
        planner = OraclePlannerRL(env)  # Original 3 m circular path.
        controller = RLController(
            env, planner, config=config.training, checkpoint=checkpoint,
        )
        controller.control()

        # The controller handles requested video and live visualization.
        if args.log:
            env.logger.save_log()
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Policy or adaptation checkpoint; otherwise use the configured default run",
    )
    run(get_args(parser))
