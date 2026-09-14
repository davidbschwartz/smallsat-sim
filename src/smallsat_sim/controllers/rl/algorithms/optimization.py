"""Shared value fitting and diagnostics; policy objectives live with PPO and VPG."""

from typing import NamedTuple

from flax import nnx
import jax
import jax.numpy as jnp


class UpdateMetrics(NamedTuple):
    actor_loss_mean: jax.Array
    critic_loss_mean: jax.Array
    true_kl_mean: jax.Array
    clip_fraction: jax.Array
    actor_steps: jax.Array
    critic_steps: jax.Array


def _diag_gaussian_kl(mu_old, std_old, mu_new, std_new):
    return jnp.sum(
        jnp.log(std_new / std_old)
        + (std_old**2 + (mu_old - mu_new) ** 2) / (2 * std_new**2)
        - 0.5,
        axis=-1,
    )


def value_loss(critic, batch):
    return jnp.mean((critic(batch.observations) - batch.returns) ** 2)


def fit_critic(graph, state, batch, key, epochs, num_minibatches):
    """Fit rewards-to-go with a separate optimizer and shuffled minibatches."""
    batch_size = batch.returns.shape[0]
    zero = jnp.array(0.0, batch.returns.dtype)

    def critic_epoch(carry, key):
        state, total = carry
        indices = jax.random.permutation(key, batch_size).reshape(num_minibatches, -1)

        def critic_step(carry, indices):
            state, total = carry
            actor, actor_opt, critic, optimizer = nnx.merge(graph, state)
            samples = jax.tree.map(lambda x: x[indices], batch)
            loss, grads = nnx.value_and_grad(lambda m: value_loss(m, samples))(critic)
            optimizer.update(grads)
            return (
                nnx.state((actor, actor_opt, critic, optimizer)),
                total + loss,
            ), None

        return jax.lax.scan(critic_step, carry, indices)[0], None

    (
        state,
        critic_sum,
    ), _ = jax.lax.scan(critic_epoch, (state, zero), jax.random.split(key, epochs))
    return state, critic_sum / (epochs * num_minibatches)
