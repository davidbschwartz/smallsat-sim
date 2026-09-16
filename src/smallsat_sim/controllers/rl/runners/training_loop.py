"""Policy training with fixed-shape collection and explicit checkpoint lifecycle."""

import time

import jax
import jax.numpy as jnp

from ..storage.rollout_batch import make_training_batch
from .runner_metrics import report, rollout_metrics


def learn_runner(runner, *, mode="fresh"):
    start_epoch = runner.begin("policy", mode)
    if mode == "fresh" and runner.training_cfg.use_pretrained:
        runner.restore("pretraining")
    collector = runner.collector("privileged", stochastic=True)
    for epoch in range(start_epoch, runner.agent.epochs):
        start = time.perf_counter()
        result = runner.collect(collector, randomize=runner.env.train_with_failures)
        batch = make_training_batch(
            result,
            runner.context_scale,
            runner.agent.gamma,
            runner.agent.lam,
        )
        metrics = runner.agent.update(batch)._asdict()
        variance = jnp.var(batch.returns)
        metrics["explained_variance"] = jnp.where(
            variance > 1e-8,
            1
            - jnp.var(batch.returns - result.values.reshape(-1))
            / jnp.maximum(variance, 1e-8),
            0.0,
        )
        metrics.update(rollout_metrics(result))
        jax.block_until_ready(metrics)
        metrics["epoch_seconds"] = time.perf_counter() - start
        report(runner, "policy_training", epoch + 1, metrics)
        runner.policy_epoch = epoch + 1
        if (epoch + 1) % runner.training_cfg.training_checkpoint_interval == 0:
            runner.save("policy")
    runner.save("policy")
