"""Vanilla policy gradient: exactly one full-rollout actor update."""

from functools import partial

from flax import nnx
import jax
import jax.numpy as jnp

from .base_agent import BaseAgent
from .optimization import UpdateMetrics, _diag_gaussian_kl, fit_critic


def vpg_loss(actor, batch):
    return -(
        actor.log_prob(batch.observations, batch.latent_actions) * batch.advantages
    ).mean()


@partial(jax.jit, static_argnames=("graph", "critic_epochs", "num_minibatches"))
def update_vpg(graph, state, batch, key, *, critic_epochs, num_minibatches):
    actor, optimizer, critic, critic_optimizer = nnx.merge(graph, state)
    old_mean, old_log_std = actor.distribution_parameters(batch.observations)

    # One gradient over the entire rollout; there is no actor epoch loop.
    actor_loss, gradients = nnx.value_and_grad(vpg_loss)(actor, batch)
    optimizer.update(gradients)
    state = nnx.state((actor, optimizer, critic, critic_optimizer))

    # The value function can fit the same returns repeatedly.
    state, critic_loss = fit_critic(
        graph, state, batch, key, critic_epochs, num_minibatches
    )
    actor, _, _, _ = nnx.merge(graph, state)
    mean, log_std = actor.distribution_parameters(batch.observations)
    kl = _diag_gaussian_kl(
        old_mean, jnp.exp(old_log_std), mean, jnp.exp(log_std)
    ).mean()
    return state, UpdateMetrics(
        actor_loss,
        critic_loss,
        kl,
        jnp.array(0.0),
        jnp.array(1),
        jnp.array(critic_epochs * num_minibatches),
    )


class VPG(BaseAgent):
    """Use the vanilla policy-gradient objective with separate value fitting."""

    algorithm = "vpg"

    def update_parameters(self, graph, state, batch, key):
        return update_vpg(
            graph,
            state,
            batch,
            key,
            critic_epochs=self.critic_training_epochs,
            num_minibatches=self.num_minibatches,
        )
