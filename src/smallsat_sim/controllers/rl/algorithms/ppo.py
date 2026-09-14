"""PPO-Clip: several shuffled policy epochs, with KL stopping between epochs."""

from functools import partial

from flax import nnx
import jax
import jax.numpy as jnp

from .base_agent import BaseAgent
from .optimization import UpdateMetrics, _diag_gaussian_kl, fit_critic


def ppo_loss(actor, batch, clip_ratio):
    ratio = jnp.exp(
        actor.log_prob(batch.observations, batch.latent_actions) - batch.logp
    )
    unclipped = ratio * batch.advantages
    clipped = jnp.clip(ratio, 1 - clip_ratio, 1 + clip_ratio) * batch.advantages
    return -jnp.minimum(unclipped, clipped).mean()


@partial(
    jax.jit,
    static_argnames=("graph", "actor_epochs", "critic_epochs", "num_minibatches"),
)
def update_ppo(
    graph,
    state,
    batch,
    key,
    *,
    actor_epochs,
    critic_epochs,
    num_minibatches,
    clip_ratio,
    target_kl,
):
    # The rollout policy defines the fixed KL reference for every actor epoch.
    actor, _, _, _ = nnx.merge(graph, state)
    old_mean, old_log_std = actor.distribution_parameters(batch.observations)
    batch_size = batch.returns.shape[0]
    zero = jnp.array(0.0, batch.returns.dtype)

    def actor_step(carry, indices):
        state, loss_sum, count = carry
        actor, optimizer, critic, critic_optimizer = nnx.merge(graph, state)
        samples = jax.tree.map(lambda x: x[indices], batch)
        loss, grads = nnx.value_and_grad(
            lambda model: ppo_loss(model, samples, clip_ratio)
        )(actor)
        optimizer.update(grads)
        state = nnx.state((actor, optimizer, critic, critic_optimizer))
        return (state, loss_sum + loss, count + 1), None

    def actor_epoch(carry, key):
        def update_epoch(carry):
            state, _, loss_sum, count = carry
            indices = jax.random.permutation(key, batch_size).reshape(
                num_minibatches, -1
            )
            (state, loss_sum, count), _ = jax.lax.scan(
                actor_step, (state, loss_sum, count), indices
            )
            actor, _, _, _ = nnx.merge(graph, state)
            mean, log_std = actor.distribution_parameters(batch.observations)
            kl = _diag_gaussian_kl(
                old_mean, jnp.exp(old_log_std), mean, jnp.exp(log_std)
            ).mean()
            return state, kl > 1.5 * target_kl, loss_sum, count

        # Once stopped, skip the entire epoch, including permutation and KL inference.
        return jax.lax.cond(carry[1], lambda state: state, update_epoch, carry), None

    actor_key, key = jax.random.split(key)
    (state, _, actor_sum, actor_count), _ = jax.lax.scan(
        actor_epoch,
        (state, jnp.array(False), zero, jnp.array(0)),
        jax.random.split(actor_key, actor_epochs),
    )

    # Critic fitting continues even when KL stopping has frozen the actor.
    state, critic_loss = fit_critic(
        graph, state, batch, key, critic_epochs, num_minibatches
    )
    actor, _, _, _ = nnx.merge(graph, state)
    mean, log_std = actor.distribution_parameters(batch.observations)
    kl = _diag_gaussian_kl(
        old_mean, jnp.exp(old_log_std), mean, jnp.exp(log_std)
    ).mean()
    ratio = jnp.exp(
        actor.log_prob(batch.observations, batch.latent_actions) - batch.logp
    )
    metrics = UpdateMetrics(
        actor_sum / jnp.maximum(actor_count, 1),
        critic_loss,
        kl,
        (jnp.abs(ratio - 1) > clip_ratio).mean(),
        actor_count,
        jnp.array(critic_epochs * num_minibatches),
    )
    return state, metrics


class PPO(BaseAgent):
    """Own the PPO update schedule; collection and model ownership are shared."""

    algorithm = "ppo"

    def __init__(self, env, planner, rng_key, activation=nnx.tanh, *, config):
        hp = config.PPO
        self.actor_training_epochs = int(hp.actor_training_epochs)
        self.clip_ratio = float(hp.clip_ratio)
        self.target_kl = float(hp.target_kl)
        if self.actor_training_epochs < 1:
            raise ValueError("PPO actor_training_epochs must be positive")
        super().__init__(env, planner, rng_key, activation, config=config)

    def update_parameters(self, graph, state, batch, key):
        return update_ppo(
            graph,
            state,
            batch,
            key,
            actor_epochs=self.actor_training_epochs,
            critic_epochs=self.critic_training_epochs,
            num_minibatches=self.num_minibatches,
            clip_ratio=self.clip_ratio,
            target_kl=self.target_kl,
        )
