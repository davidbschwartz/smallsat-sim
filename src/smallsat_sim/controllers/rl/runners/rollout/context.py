"""Context-estimator history, warm-up and per-environment reset operations."""
from typing import NamedTuple
import jax
import jax.numpy as jnp


class ContextHistory(NamedTuple):
    """Values are [environment, time, state-and-action]; counts mark valid history."""
    values: jax.Array
    counts: jax.Array


def append_history(history, states, actions):
    state_action = jnp.concatenate((states, actions), axis=-1)
    values = jnp.concatenate(
        (history.values[:, 1:], state_action[:, None]),
        axis=1,
    )
    counts = jnp.minimum(history.counts + 1, values.shape[1])
    return ContextHistory(values=values, counts=counts)


def reset_history(history, mask):
    return ContextHistory(
        values=jnp.where(mask[:, None, None], 0, history.values),
        counts=jnp.where(mask, 0, history.counts),
    )


def estimate_context(model, history, scale):
    prediction = model(history.values) * scale
    return jnp.where(
        (history.counts >= history.values.shape[1])[:, None], prediction, 0.0
    )

