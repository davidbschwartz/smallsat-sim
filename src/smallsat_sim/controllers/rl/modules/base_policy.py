"""Diagonal Gaussian policy with a sigmoid transform to physical thrust bounds.

Training stores the Gaussian (pre-squash) sample. Never recover a PPO training
sample by inverting a rounded/saturated physical command.
"""

from typing import NamedTuple

from flax import nnx
import distrax
import jax
import jax.numpy as jnp
import numpy as np

from .base_network import normalize_hidden_sizes
from .mlp import mlp


class PolicySample(NamedTuple):
    actions: jax.Array
    latent_actions: jax.Array
    logp: jax.Array


class Actor(nnx.Module):
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes,
        activation,
        res_dim,
        act_low,
        act_high,
        initial_log_std=-0.5,
        log_std_min=None,
        *,
        rngs,
    ):
        low, high = np.asarray(act_low), np.asarray(act_high)
        if low.shape != (act_dim,) or high.shape != (act_dim,):
            raise ValueError("Action bounds must have shape (act_dim,)")
        if not np.all(np.isfinite(low) & np.isfinite(high) & (high > low)):
            raise ValueError(
                "Action bounds must be finite with strictly positive width"
            )
        self.obs_dim, self.act_dim, self.res_dim = obs_dim, act_dim, res_dim
        self.act_low, self.act_high = jnp.asarray(low), jnp.asarray(high)
        self.act_range = self.act_high - self.act_low
        self.log_std = nnx.Param(
            jnp.full((act_dim,), initial_log_std, dtype=jnp.float32)
        )
        self.log_std_min = log_std_min
        self.mu_net = mlp(
            [obs_dim + res_dim, *normalize_hidden_sizes(hidden_sizes), act_dim],
            activation,
            last_layer_std=0.01,
            rngs=rngs,
        )

    def distribution_parameters(self, observations):
        log_std = self.log_std.value
        if self.log_std_min is not None:
            log_std = jnp.maximum(log_std, self.log_std_min)
        return self.mu_net(observations), log_std

    def _distribution(self, observations):
        mean, log_std = self.distribution_parameters(observations)
        return distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))

    def log_prob(self, observations, latent_actions):
        mean, log_std = self.distribution_parameters(observations)
        return self._log_prob_diag_gaussian_pre_squash(mean, log_std, latent_actions)

    def _log_prob_diag_gaussian_pre_squash(self, mean, log_std, latent):
        normal = -0.5 * jnp.sum(
            ((latent - mean) * jnp.exp(-log_std)) ** 2
            + 2 * log_std
            + jnp.log(2 * jnp.pi),
            axis=-1,
        )
        jacobian = jnp.sum(
            jnp.log(self.act_range)
            + jax.nn.log_sigmoid(latent)
            + jax.nn.log_sigmoid(-latent),
            axis=-1,
        )
        return normal - jacobian

    def sample(self, observations, key):
        mean, log_std = self.distribution_parameters(observations)
        latent = mean + jnp.exp(log_std) * jax.random.normal(
            key, mean.shape, mean.dtype
        )
        return PolicySample(
            self.apply_action_bounds(latent),
            latent,
            self._log_prob_diag_gaussian_pre_squash(mean, log_std, latent),
        )

    def sample_action_and_logp(self, observations, key):
        sample = self.sample(observations, key)
        return sample.actions, sample.logp

    def apply_action_bounds(self, latent):
        return self.act_low + self.act_range * jax.nn.sigmoid(latent)

    def deterministic_action(self, observations):
        return self.apply_action_bounds(self.mu_net(observations))

    def forward(self, observations, latent_actions=None):
        return self._distribution(observations), (
            None
            if latent_actions is None
            else self.log_prob(observations, latent_actions)
        )
