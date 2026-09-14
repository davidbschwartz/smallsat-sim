"""Independent twin action-value networks; actions are normalized to [0, 1]."""
from flax import nnx
import jax.numpy as jnp
from .mlp import mlp


class TwinQ(nnx.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, *, rngs):
        sizes = [obs_dim + act_dim, *hidden_sizes, 1]
        self.q1 = mlp(sizes, nnx.relu, rngs=rngs)
        self.q2 = mlp(sizes, nnx.relu, rngs=rngs)

    def __call__(self, observations, actions):
        inputs = jnp.concatenate((observations, actions), axis=-1)
        return self.q1(inputs)[..., 0], self.q2(inputs)[..., 0]
