"""Evaluate explicit context sources; privileged runs are diagnostics only."""

import jax
import jax.numpy as jnp

from .adaptation_training import gather_windows
from .runner_metrics import report, rollout_metrics


def evaluate_runner(
    runner,
    phase=2,
    *,
    allow_privileged_context=False,
    context_source=None,
    randomize=None,
):
    if phase not in (1, 2):
        raise ValueError("Phase must be 1 or 2")
    source = context_source or ("privileged" if phase == 1 else "estimated")
    if runner.am is None:
        source = "zero"
    if source == "privileged" and not allow_privileged_context:
        raise ValueError("Privileged evaluation requires allow_privileged_context=True")
    runner.restore("adaptation" if source == "estimated" else "policy")
    collector = runner.collector(
        source,
        stochastic=False,
        steps=runner.training_cfg.episode_len,
        mode="evaluation",
    )
    rows = []
    for index in range(runner.training_cfg.n_evals):
        result = runner.collect(
            collector,
            randomize=(
                runner.env.train_with_failures if randomize is None else randomize
            ),
        )
        metrics = rollout_metrics(result)
        if source == "estimated":
            # Compare the post-transition estimate with that transition's label.
            features = jnp.concatenate(
                (result.step_outputs.prev_states, result.actions),
                axis=-1,
            )
            # Validation sampling stays fixed-size and avoids materializing every window.
            valid = result.labels.history_full
            flat = jnp.nonzero(
                valid.reshape(-1),
                size=runner.training_cfg.am_batch_size,
                fill_value=0,
            )[0]
            indices = jnp.stack(
                (flat // runner.env.num_envs, flat % runner.env.num_envs),
                axis=-1,
            )
            x, y = gather_windows(
                features,
                result.labels.normalized_context,
                indices,
                runner.env.history_len,
            )
            mask = valid.reshape(-1)[flat]
            errors = jnp.linalg.norm((runner.am(x) - y) * runner.context_scale, axis=-1)
            metrics["mean_extrinsic_error"] = (errors * mask).sum() / jnp.maximum(
                mask.sum(), 1
            )
        rows.append(report(runner, "evaluation_" + source, index + 1, metrics))
    return rows
