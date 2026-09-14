"""Supervised context learning on trajectories induced by the current estimator."""

from functools import partial
from pathlib import Path

from flax import nnx
import jax
import jax.numpy as jnp

from .runner_metrics import report
from .runner_utils import (
    checkpoint_exists,
    load_trained_modules,
    restore_trained_modules,
    model_fingerprint,
)


def gather_windows(features, targets, indices, history_len):
    # Indices end at the completed transition whose context is the target.
    time, env = indices[:, 0], indices[:, 1]
    offsets = jnp.arange(history_len) - history_len + 1
    return features[time[:, None] + offsets, env[:, None]], targets[time, env]


@partial(jax.jit, static_argnames=("graph", "history_len", "batch_size", "updates"))
def adaptation_updates(
    graph, state, features, targets, full, key, *, history_len, batch_size, updates
):
    """Split by environment before sampling windows; no overlapping split leakage."""
    # Separate environments before drawing overlapping history windows.
    env_count = features.shape[1]
    split = max(1, min(env_count - 1, int(0.8 * env_count)))
    env_ids = jnp.arange(env_count)[None, :]
    time_ids = jnp.arange(features.shape[0])[:, None]
    valid = full & (time_ids >= history_len - 1)
    train_mask = valid & (env_ids < split)
    val_mask = valid & (env_ids >= split)

    def draw(mask, key):
        # Fixed output shape even when episodes have different lengths.
        cumulative = jnp.cumsum(mask.reshape(-1), dtype=jnp.int32)
        ranks = jax.random.randint(
            key, (batch_size,), 0, jnp.maximum(cumulative[-1], 1)
        )
        flat = jnp.searchsorted(cumulative, ranks, side="right")
        flat = jnp.minimum(flat, mask.size - 1)
        indices = jnp.stack((flat // env_count, flat % env_count), axis=-1)
        return gather_windows(features, targets, indices, history_len)

    def loss(model, x, y):
        return jnp.square(model(x) - y).mean()

    def update(state, key):
        model, optimizer = nnx.merge(graph, state)
        x, y = draw(train_mask, key)
        value, grads = nnx.value_and_grad(loss)(model, x, y)
        optimizer.update(grads)
        return nnx.state((model, optimizer)), value

    train_key, val_key = jax.random.split(key)
    state, losses = jax.lax.cond(
        jnp.any(train_mask),
        lambda s: jax.lax.scan(update, s, jax.random.split(train_key, updates)),
        lambda s: (s, jnp.zeros((updates,))),
        state,
    )
    model, _ = nnx.merge(graph, state)
    x, y = draw(val_mask, val_key)
    val_loss = jnp.where(jnp.any(val_mask), loss(model, x, y), jnp.nan)
    return state, {
        "am_train_loss_mean": losses.mean(),
        "am_val_loss_mean": val_loss,
        "train_windows": train_mask.sum(),
        "validation_windows": val_mask.sum(),
    }


def train_adaptation_module_on_policy_runner(runner, *, mode="fresh"):
    if runner.am is None:
        raise ValueError("Adaptation requires an adaptive PPO run")
    if runner.env.num_envs < 2:
        raise ValueError(
            "Adaptation requires at least two environments for train/validation separation"
        )
    if runner.agent.steps_per_epoch < runner.env.history_len:
        raise ValueError("Rollout must contain at least one full history")

    # The policy is fixed throughout this stage; only estimator weights change.
    start_epoch = runner.begin("adaptation", mode)
    collector = runner.collector("estimated", stochastic=False)
    for epoch in range(start_epoch, runner.training_cfg.am_collection_epochs):
        result = runner.collect(collector, randomize=runner.env.train_with_failures)
        features = jnp.concatenate(
            (result.step_outputs.prev_states, result.actions), axis=-1
        )
        objects = (runner.am, runner.am_optimizer)
        graph, state = nnx.split(objects)
        state, metrics = adaptation_updates(
            graph,
            state,
            features,
            result.labels.normalized_context,
            result.labels.history_full,
            runner._take_keys(),
            history_len=runner.env.history_len,
            batch_size=runner.training_cfg.am_batch_size,
            updates=runner.training_cfg.am_epochs,
        )
        nnx.update(objects, state)
        values = report(runner, "am_training", epoch + 1, metrics)
        if values["train_windows"] == 0 or values["validation_windows"] == 0:
            raise ValueError(
                "No complete adaptation histories in a split; increase the horizon or shorten history"
            )
        runner.adaptation_epoch = epoch + 1
        if (epoch + 1) % runner.training_cfg.am_checkpoint_interval == 0:
            runner.save("adaptation")
    runner.save("adaptation")


def initialize_from_teacher(runner, checkpoint):
    """Copy a fixed teacher into a new adaptation run; ordinary resume stays strict."""
    if runner.am is None or runner.agent.algorithm != "ppo":
        raise ValueError("A shared teacher requires an adaptive PPO run")
    for stage in ("policy", "adaptation"):
        if checkpoint_exists(runner.ckpt_dir, runner._filename(stage)):
            raise FileExistsError(
                "Initialize a teacher only in a new run directory/name"
            )

    checkpoint = Path(checkpoint).resolve()
    payload = load_trained_modules(checkpoint.parent, checkpoint.name)
    metadata = payload["metadata"]
    if metadata.get("stage") != "policy":
        raise ValueError("The teacher must be a policy-stage checkpoint")

    # Estimator choices and output locations may differ; policy, physics and task may not.
    estimator_settings = {
        "am_architecture",
        "context_window_len",
        "am_epochs",
        "am_collection_epochs",
        "am_lr",
        "am_batch_size",
        "am_checkpoint_interval",
        "checkpoint_dir",
        "rl_run_id",
    }

    def teacher_contract(config):
        return {
            **config,
            "rl": {
                name: value
                for name, value in config["rl"].items()
                if name not in estimator_settings
            },
        }

    if teacher_contract(metadata["resolved_config"]) != teacher_contract(
        runner.resolved_config
    ):
        raise ValueError(
            "Teacher policy, physics or task configuration differs from this run"
        )

    # Copy policy state, not the source estimator; the destination estimator stays fresh.
    restore_trained_modules(runner.agent, checkpoint.parent, checkpoint.name)
    runner._rng = metadata["runner_rng"]
    runner.env._rng = metadata["env_rng"]
    runner.policy_epoch = int(metadata["policy_epoch"])
    runner.adaptation_epoch = 0
    runner.teacher_source = {
        "path": str(checkpoint),
        "config_id": metadata["config_id"],
        "model_id": model_fingerprint(runner.agent.actor),
    }
    # A local policy checkpoint makes evaluation/resume independent of the source path.
    runner.save("policy")
