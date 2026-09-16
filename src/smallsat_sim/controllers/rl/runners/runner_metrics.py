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


def scenario_metrics(output, actions, trajectory, *, initial_poses, dt):
    """Score the first episode in each lane using the environment's terminal flags.

    Later scan samples never affect a finished episode. Nonfinite transitions
    fail the episode and are excluded from metrics and recordings.
    """
    import numpy as np

    poses, velocities = trajectory
    steps, lanes = actions.shape[:2]
    valid_control = np.isfinite(actions).all(axis=-1)
    valid_state = np.isfinite(poses).all(axis=-1) & np.isfinite(velocities).all(axis=-1)
    valid_state &= np.linalg.norm(poses[..., 3:7], axis=-1) > 0
    valid = valid_control & valid_state
    ended = output.terminals.astype(bool) | ~valid
    first = np.where(ended.any(axis=0), ended.argmax(axis=0), steps)
    positions = np.linalg.norm(output.next_position_error, axis=-1)
    results = []
    for lane in range(lanes):
        end = int(first[lane])
        reason, success, n = "timeout", False, steps
        if end < steps:
            n = end + int(valid[end, lane])
            if not valid_control[end, lane]:
                reason = "nonfinite_control"
            elif not valid_state[end, lane]:
                reason = "nonfinite_state"
            elif output.failure_terminals[end, lane]:
                reason = "failure"
            elif output.success_terminals[end, lane]:
                reason, success = "success", True
            else:
                reason = "terminated"
        row = dict(
            success=success, completion_fraction=float(success),
            completion_time=n * dt if success else None,
            episode_length=n, termination_reason=reason,
            episodic_return=float(output.rewards[:n, lane].sum()),
            control_effort=float(np.abs(actions[:n, lane]).sum() * dt),
        )
        for name, values in (("position_error", positions[:n, lane]),
                             ("attitude_error", output.next_attitude_error[:n, lane])):
            row.update({f"mean_{name}": float(values.mean()) if n else None,
                        f"max_{name}": float(values.max()) if n else None,
                        f"final_{name}": float(values[-1]) if n else None})
        results.append(dict(metrics=row, time=np.arange(n + 1) * dt,
                            qpos=np.concatenate([initial_poses[lane:lane+1], poses[:n, lane]])))
    return results
