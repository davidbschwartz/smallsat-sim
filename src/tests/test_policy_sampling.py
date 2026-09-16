"""Policy sampling bounds and distribution behavior tests."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from smallsat_sim.controllers.rl.modules.base_policy import Actor
from smallsat_sim.controllers.rl.modules.mlp import mlp


def actor(seed=0):
    return Actor(3, 2, [4, 4], nnx.tanh, 0, jnp.array([-2., 0.]), jnp.array([2., 1.5]), rngs=nnx.Rngs(seed))


def test_sample_likelihood_including_saturation():
    model = actor()
    obs = jnp.zeros((4, 3))
    latent = jnp.array([[0., 0.], [20., -20.], [100., -100.], [-4., 5.]])
    old_logp = model.log_prob(obs, latent)
    mean, log_std = model.distribution_parameters(obs)
    reference = -0.5 * jnp.sum(((latent-mean)/jnp.exp(log_std))**2 + 2*log_std + jnp.log(2*jnp.pi), axis=-1)
    reference -= jnp.sum(jnp.log(model.act_range) - jax.nn.softplus(-latent) - jax.nn.softplus(latent), axis=-1)
    np.testing.assert_allclose(old_logp, reference, atol=1e-5, rtol=1e-6)
    grads = nnx.grad(lambda m: m.log_prob(obs, latent).mean())(model)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(grads))
    sample = model.sample(obs, jax.random.PRNGKey(1))
    np.testing.assert_allclose(sample.logp, model.log_prob(obs, sample.latent_actions))
    assert jnp.all((sample.actions >= model.act_low) & (sample.actions <= model.act_high))


def test_model_initialization_is_seeded_without_duplicate_layers():
    first, same, other = actor(1), actor(1), actor(2)
    for a, b in zip(jax.tree.leaves(nnx.state(first)), jax.tree.leaves(nnx.state(same))):
        np.testing.assert_array_equal(a, b)
    assert not np.array_equal(first.mu_net.layers[0].kernel, other.mu_net.layers[0].kernel)
    network = mlp([4, 4, 4, 4], rngs=nnx.Rngs(1))
    assert not np.array_equal(network.layers[0].kernel, network.layers[2].kernel)
    assert set(nnx.state(first, nnx.Param)) == {"log_std", "mu_net"}


def test_invalid_action_bounds_rejected():
    with pytest.raises(ValueError, match="positive width"):
        Actor(3, 2, [4], nnx.tanh, 0, jnp.zeros(2), jnp.zeros(2), rngs=nnx.Rngs(0))
