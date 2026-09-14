"""Small end-to-end RL runner smoke tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from smallsat_sim.controllers.rl.runners.on_policy_runner import OnPolicyRunner
from smallsat_sim.envs.vehicles.astrobee_rl import config as rl_config
from smallsat_sim.envs.vehicles.astrobee_rl.env import AstrobeeEnvVectorized
from smallsat_sim.planners.oracle.oracle_rl import OraclePlannerRL


def test_nominal_learn_smoke_emits_finite_metrics(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")

    # Keep smoke run tiny for CI/local validation speed.
    config = rl_config.resolve_config(
        num_envs=16, train_with_failures=False, use_pretrained=False,
        use_adaptive_approach=False,
    )

    env = None
    try:
        args = SimpleNamespace(
            headless=True,
            num_bodies=1,
            video=False,
            log=True,
            wandb=False,
        )
        env = AstrobeeEnvVectorized(
            args=args,
            run_name="smoke_nominal",
            config=config.env,
        )
        cfg = config.training
        ppo = cfg.PPO
        ppo.steps_per_epoch = 16
        ppo.epochs = 2
        ppo.max_ep_len = 16
        ppo.actor_training_epochs = 1
        ppo.critic_training_epochs = 1
        ppo.num_minibatches = 2
        cfg.episode_len = 16
        cfg.n_evals = 1

        planner = OraclePlannerRL(env, radius=0.0)
        runner = OnPolicyRunner(env, planner, config=config.training)
        runner.ckpt_dir = str(tmp_path / "checkpoints")

        runner.learn()

        entries = list(env.logger.logs.values())
        policy_entries = [
            e for e in entries if e.get("stage") == "policy_training"
        ]
        assert len(policy_entries) == ppo.epochs

        keys_to_check = [
            "actor_loss_mean",
            "critic_loss_mean",
            "true_kl_mean",
            "clip_fraction",
            "explained_variance",
            "mean_lateral_error",
            "mean_angle_error",
            "mean_final_position_error",
            "success_rate",
            "terminated_step_count",
            "success_termination_step_count",
            "failure_termination_step_count",
        ]

        for entry in policy_entries:
            for key in keys_to_check:
                value = entry[key]
                assert np.isfinite(float(value)), f"Metric {key} is non-finite: {value}"
    finally:
        if env is not None:
            env.close()
