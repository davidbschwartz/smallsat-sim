"""Temporal convolution on [history, channels], with one context output."""

from flax import nnx
import jax.numpy as jnp


class CNNAdaptationModule(nnx.Module):
    def __init__(self, n_steps, state_action_dim, ext_dim, *, rngs):
        if min(n_steps, state_action_dim, ext_dim) <= 0:
            raise ValueError("History and feature dimensions must be positive")
        self.n_steps, self.state_action_dim = n_steps, state_action_dim
        self.encoder = nnx.Linear(state_action_dim, 32, rngs=rngs)
        self.conv1 = nnx.Conv(32, 32, kernel_size=(5,), strides=(2,), rngs=rngs)
        self.conv2 = nnx.Conv(32, 32, kernel_size=(3,), strides=(2,), rngs=rngs)
        self.output = nnx.Linear(((n_steps + 3) // 4) * 32, ext_dim, rngs=rngs)

    def __call__(self, history):
        if history.shape[-2:] != (self.n_steps, self.state_action_dim):
            raise ValueError("Expected (..., history_length, state_action_dim) history")
        x = nnx.relu(self.encoder(history))
        x = nnx.relu(self.conv1(x))
        x = nnx.relu(self.conv2(x))
        return self.output(x.reshape((*x.shape[:-2], -1)))
