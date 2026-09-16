"""Evaluate PD on the same setpoint/scenario protocol as the learned policies."""

from pathlib import Path
from types import SimpleNamespace
import argparse
import json

import jax

from .evaluation import CORE, STRESS, prepare
from smallsat_sim.configuration import read_yaml
from smallsat_sim.controllers.rl.runners.pretraining import pd_demonstrator
from smallsat_sim.controllers.rl.runners.rollout.collector import make_collector
from smallsat_sim.controllers.rl.runners.runner_metrics import episode_metrics
from smallsat_sim.envs.vehicles.astrobee_rl.env import AstrobeeEnvVectorized
from smallsat_sim.envs.vehicles.astrobee_rl.config import resolve_config
from smallsat_sim.planners.oracle.oracle_rl import OraclePlannerRL


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--suite", choices=("core", "stress"), default="core")
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/rl_results/pd.json")
    )
    args = parser.parse_args(argv)
    overrides = read_yaml(args.config) if args.config else {}
    config = resolve_config(
        use_adaptive_approach=False,
        overrides=overrides,
    )
    env = AstrobeeEnvVectorized(
        SimpleNamespace(headless=True, video=False, log=False, wandb=False),
        run_name="pd", config=config.env,
    )
    try:
        planner = OraclePlannerRL(env, radius=0.0)
        reference = planner.get_reference(env.get_obs())
        collector = make_collector(
            env,
            None,
            None,
            steps=config.training.episode_len,
            context_source="zero",
            stochastic=False,
            demonstration=pd_demonstrator(
                env,
                planner,
                reference,
            ),
        )
        rows = []
        for index, scenario in enumerate(
            CORE + (STRESS if args.suite == "stress" else ())
        ):
            seed = args.seed + index
            state = prepare(env, scenario, seed)
            if env.env_cfg.environment.rollout_backend == "freeflyer":
                state = env.freeflyer_state_struct()
            result = collector(
                state, None, None, None, jax.random.PRNGKey(seed), reference
            )
            metrics = episode_metrics(
                result,
                dt=env.model.opt.timestep * env.env_cfg.environment.control_decimation,
                onset_seconds=scenario.onset_seconds,
            )
            rows.append(
                {
                    "experiment": "pd",
                    "training_seed": 0,
                    "seed": seed,
                    "scenario": scenario.name,
                    "context_source": "none",
                    "backend": env.env_cfg.environment.rollout_backend,
                    **{
                        key: float(value)
                        for key, value in jax.device_get(metrics).items()
                    },
                }
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
