"""Fixed-capacity device replay. Capacity counts individual transitions."""
from typing import NamedTuple
from functools import partial
import jax
import jax.numpy as jnp


class Transition(NamedTuple):
    observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    next_observations: jax.Array
    terminated: jax.Array
    truncated: jax.Array


class ReplayState(NamedTuple):
    data: Transition
    cursor: jax.Array
    size: jax.Array


def empty_replay(capacity, obs_dim, act_dim):
    if min(capacity, obs_dim, act_dim) < 1:
        raise ValueError("Replay dimensions must be positive")
    return ReplayState(Transition(
        jnp.zeros((capacity, obs_dim), jnp.float32), jnp.zeros((capacity, act_dim), jnp.float32),
        jnp.zeros(capacity, jnp.float32), jnp.zeros((capacity, obs_dim), jnp.float32),
        jnp.zeros(capacity, dtype=bool), jnp.zeros(capacity, dtype=bool),
    ), jnp.array(0, jnp.int32), jnp.array(0, jnp.int32))


@jax.jit
def insert(replay, transitions):
    capacity = replay.data.rewards.shape[0]
    count = transitions.rewards.shape[0]
    if count > capacity:
        raise ValueError("Collection chunk exceeds replay capacity")
    indices = (replay.cursor + jnp.arange(count)) % capacity
    data = jax.tree.map(lambda x, y: x.at[indices].set(y), replay.data, transitions)
    return ReplayState(data, (replay.cursor + count) % capacity,
                       jnp.minimum(capacity, replay.size + count))


@partial(jax.jit, static_argnames=('batch_size',))
def sample(replay, key, batch_size):
    # The runner guarantees size > 0 before sampling. Sampling is with replacement.
    indices = jax.random.randint(key, (batch_size,), 0, replay.size)
    return jax.tree.map(lambda x: x[indices], replay.data)
