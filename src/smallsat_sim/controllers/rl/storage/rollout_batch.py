"""On-policy data and masked, bootstrapped return calculations."""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class TrainingBatch(NamedTuple):
    observations: jax.Array
    latent_actions: jax.Array
    logp: jax.Array
    advantages: jax.Array
    returns: jax.Array


def normalize_advantages(advantages):
    return (advantages - advantages.mean()) / (advantages.std() + 1e-8)


@jax.jit
def advantages_and_returns(rewards, values, done, bootstrap, gamma, lam):
    """Each done transition uses its own pre-reset bootstrap (zero if terminal).

    Bootstrap[-1] also contains the next value at a nonterminal rollout cutoff.
    Critic targets are discounted rewards-to-go, not GAE + values.
    """

    def step(carry, sample):
        next_adv, next_return, next_value = carry
        reward, value, boundary, boot = sample
        continuation = (~boundary).astype(value.dtype)
        next_value = jnp.where(boundary, boot, next_value)
        delta = reward + gamma * next_value - value
        advantage = delta + gamma * lam * continuation * next_adv
        target = reward + gamma * jnp.where(boundary, boot, next_return)
        return (advantage, target, value), (advantage, target)

    initial = (jnp.zeros_like(values[-1]), bootstrap[-1], bootstrap[-1])
    _, result = jax.lax.scan(
        step, initial, (rewards, values, done, bootstrap), reverse=True
    )
    return jax.tree.map(jax.lax.stop_gradient, result)


def make_training_batch(result, scale, gamma, lam):
    """Build return targets, then flatten time/environment axes for learning."""
    advantages, returns = advantages_and_returns(
        result.step_outputs.rewards,
        result.values,
        result.done_masks,
        result.bootstrap_values,
        gamma,
        lam,
    )
    observations = jnp.concatenate(
        (result.step_outputs.prev_states, result.context / scale),
        axis=-1,
    )
    batch = TrainingBatch(
        observations=observations,
        latent_actions=result.labels.latent_actions,
        logp=result.logp,
        advantages=advantages,
        returns=returns,
    )
    return jax.tree.map(lambda x: x.reshape((-1, *x.shape[2:])), batch)
