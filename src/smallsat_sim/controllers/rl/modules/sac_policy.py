"""SAC policy. Entropy is measured on normalized actions in [0, 1]."""
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from .mlp import mlp
from .base_policy import PolicySample


class SACActor(nnx.Module):
    def __init__(self, obs_dim, act_low, act_high, hidden_sizes, *, rngs,
                 log_std_min=-5.0, log_std_max=2.0):
        low, high = np.asarray(act_low), np.asarray(act_high)
        if low.ndim != 1 or high.shape != low.shape or not np.all(
            np.isfinite(low) & np.isfinite(high) & (high > low)
        ):
            raise ValueError("Action bounds must be finite vectors with positive width")
        self.act_low, self.act_high = jnp.asarray(low), jnp.asarray(high)
        self.act_range = self.act_high - self.act_low
        self.log_std_min, self.log_std_max = log_std_min, log_std_max
        self.net = mlp([obs_dim, *hidden_sizes, 2 * len(low)], nnx.relu, rngs=rngs)

    def distribution_parameters(self, observations):
        mean, log_std = jnp.split(self.net(observations), 2, axis=-1)
        return mean, jnp.clip(log_std, self.log_std_min, self.log_std_max)

    def apply_action_bounds(self, latent):
        return self.act_low + self.act_range * jax.nn.sigmoid(latent)

    def normalize_action(self, actions):
        return (actions - self.act_low) / self.act_range

    def sample(self, observations, key):
        mean, log_std = self.distribution_parameters(observations)
        noise = jax.random.normal(key, mean.shape, dtype=mean.dtype)
        latent = mean + jnp.exp(log_std) * noise
        normal = -0.5 * (noise**2 + 2 * log_std + jnp.log(2 * jnp.pi))
        jacobian = jax.nn.log_sigmoid(latent) + jax.nn.log_sigmoid(-latent)
        return PolicySample(self.apply_action_bounds(latent), latent,
                            (normal - jacobian).sum(axis=-1))

    def deterministic_action(self, observations):
        mean, _ = self.distribution_parameters(observations)
        return self.apply_action_bounds(mean)
