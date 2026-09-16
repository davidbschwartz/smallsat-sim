"""Shared neural-network helpers for RL modules."""

from flax import nnx

from .mlp import mlp


def normalize_hidden_sizes(hidden_sizes):
    sizes = [hidden_sizes] if isinstance(hidden_sizes, int) else list(hidden_sizes)
    if any(width <= 0 for width in sizes):
        raise ValueError("Hidden layer widths must be positive")
    return sizes


class Critic(nnx.Module):
    def __init__(self, obs_dim, hidden_sizes, activation, res_dim, *, rngs):
        self.v_net = mlp(
            [obs_dim + res_dim, *normalize_hidden_sizes(hidden_sizes), 1],
            activation,
            rngs=rngs,
        )

    def forward(self, observations):
        return self.v_net(observations)[..., 0]

    __call__ = forward
