"""Train and evaluate the default Astrobee RL workflow."""

import wandb

from smallsat_sim.controllers.rl.runners.factory import make_runner
from smallsat_sim.envs.vehicles.astrobee_rl.env import AstrobeeEnvVectorized
from smallsat_sim.envs.vehicles.astrobee_rl.config import resolve_config
from smallsat_sim.planners.oracle.oracle_rl import OraclePlannerRL
from smallsat_sim.utils.helpers import get_args


def main():
    args = get_args()
    config = resolve_config()
    env = AstrobeeEnvVectorized(args=args, config=config.env)
    try:
        planner = OraclePlannerRL(env, radius=0.0)
        runner = make_runner(env, planner, config=config.training)

        # Edit the stages here. To resume RL, omit pretraining and pass
        # mode="resume" to learn (and to adaptation if that stage has started).
        if config.training.algorithm != "sac":
            runner.pretrain()
        runner.learn()
        if env.use_adaptive_approach:
            runner.train_adaptation_module_on_policy()
        runner.evaluate()

        # The runner handles requested rollout videos and live visualization.
        if args.log:
            env.logger.save_log()
    finally:
        env.close()
        if args.wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
