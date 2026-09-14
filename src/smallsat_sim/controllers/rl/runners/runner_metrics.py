"""Device reductions first; one host transfer per report."""

from pathlib import Path
import json

import jax
import jax.numpy as jnp
import wandb


def rollout_metrics(result):
    output = result.step_outputs
    completed = result.done_masks.sum()
    successes = output.success_terminals.sum()
    return {
        "mean_episodic_returns": result.episode_returns.sum()
        / jnp.maximum(completed, 1),
        "success_rate": successes / jnp.maximum(completed, 1),
        "terminated_step_count": output.terminals.sum(),
        "success_termination_step_count": successes,
        "failure_termination_step_count": output.failure_terminals.sum(),
        "timeout_count": result.truncated_masks.sum(),
        "completed_episode_count": completed,
        "mean_lateral_error": jnp.linalg.norm(
            output.next_position_error, axis=-1
        ).mean(),
        "mean_angle_error": jnp.rad2deg(output.next_attitude_error).mean(),
        "mean_final_position_error": jnp.linalg.norm(
            output.next_position_error[-1],
            axis=-1,
        ).mean(),
        "mean_reward": output.rewards.mean(),
        "control_effort": jnp.square(result.actions).sum(axis=-1).mean(),
    }


def report(runner, stage, epoch, metrics):
    values = {name: float(value) for name, value in jax.device_get(metrics).items()}
    path = (
        Path(runner.ckpt_dir).parent
        / "metrics"
        / f"{runner.env.run_name}_seed{runner.env.env_cfg.sim.seed}.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(
            json.dumps(
                {
                    "experiment": runner.env.run_name,
                    "seed": runner.env.env_cfg.sim.seed,
                    "stage": stage,
                    "epoch": epoch,
                    **values,
                }
            )
            + "\n"
        )
    if runner.agent.has_logger:
        runner.env.logger.log(
            runner.env.run_id,
            float(epoch),
            run_name=runner.env.run_name,
            stage=stage,
            **values,
        )
    if runner.env.use_wandb:
        wandb.log({f"{stage}/{key}": value for key, value in values.items()})
    print(
        f"{stage} {epoch}: "
        + ", ".join(
            f"{key}={value:.4g}"
            for key, value in values.items()
            if key
            in (
                "actor_loss_mean",
                "critic_loss_mean",
                "am_train_loss_mean",
                "success_rate",
            )
        ),
        flush=True,
    )
    return values


def episode_metrics(result, *, dt, onset_seconds):
    """One initial episode per environment, excluding all post-reset samples."""
    output = result.step_outputs
    prior_done = jnp.concatenate(
        (
            jnp.zeros_like(result.done_masks[:1], dtype=jnp.int32),
            jnp.cumsum(result.done_masks[:-1], axis=0),
        ),
        axis=0,
    )
    valid = prior_done == 0
    success = output.success_terminals & valid
    failure = output.failure_terminals & valid
    successes = success.any(axis=0)
    failures = failure.any(axis=0)
    denominator = jnp.maximum(valid.sum(), 1)
    times = (jnp.arange(valid.shape[0]) + 1)[:, None] * dt
    exposed = (valid & (times > onset_seconds)).any(axis=0)
    recovered = (success & (times > onset_seconds)).any(axis=0)
    recovery_times = jnp.where(
        success & (times > onset_seconds),
        times - onset_seconds,
        0.0,
    ).sum(axis=0)
    return {
        "success_rate": successes.mean(),
        "failure_rate": failures.mean(),
        "timeout_rate": (~(successes | failures)).mean(),
        "mean_episodic_returns": (output.rewards * valid).sum(axis=0).mean(),
        "mean_lateral_error": (
            jnp.linalg.norm(output.next_position_error, axis=-1) * valid
        ).sum()
        / denominator,
        "mean_angle_error": (jnp.rad2deg(output.next_attitude_error) * valid).sum()
        / denominator,
        "control_effort": (jnp.square(result.actions).sum(axis=-1) * valid).sum()
        / denominator,
        "fault_exposed_envs": exposed.sum(),
        "recovered_envs": recovered.sum(),
        "recovery_fraction": recovered.sum() / jnp.maximum(exposed.sum(), 1),
        # Interpret only alongside recovered_envs; zero means no measured recoveries when count=0.
        "mean_recovery_seconds": recovery_times.sum() / jnp.maximum(recovered.sum(), 1),
    }
