"""On-policy PPO and VPG update math tests."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from smallsat_sim.controllers.rl.modules.base_policy import Actor
from smallsat_sim.controllers.rl.modules.base_network import Critic
from smallsat_sim.controllers.rl.algorithms.ppo import ppo_loss, update_ppo
from smallsat_sim.controllers.rl.algorithms.vpg import vpg_loss, update_vpg
from smallsat_sim.controllers.rl.storage.rollout_batch import (
    TrainingBatch,
    advantages_and_returns,
)


def fixture():
    actor = Actor(2, 1, [4], nnx.tanh, 0, jnp.zeros(1), jnp.ones(1), rngs=nnx.Rngs(1))
    critic = Critic(2, [4], nnx.tanh, 0, rngs=nnx.Rngs(2))
    obs = jnp.arange(8, dtype=jnp.float32).reshape(4, 2) / 10
    sample = actor.sample(obs, jax.random.PRNGKey(3))
    batch = TrainingBatch(
        obs,
        sample.latent_actions,
        sample.logp,
        jnp.array([-1.0, -0.5, 0.5, 1.0]),
        jnp.ones(4),
    )
    return actor, critic, batch


def test_vpg_matches_one_reference_optimizer_step():
    actor, critic, batch = fixture()
    actor_opt = nnx.Optimizer(actor, optax.adam(0.01))
    critic_opt = nnx.Optimizer(critic, optax.adam(0.01))
    graph, state = nnx.split((actor, actor_opt, critic, critic_opt))
    expected, opt, _, _ = nnx.merge(graph, state)
    loss, grads = nnx.value_and_grad(
        lambda m: -(
            m.log_prob(batch.observations, batch.latent_actions) * batch.advantages
        ).mean()
    )(expected)
    opt.update(grads)
    actual_state, metrics = update_vpg(
        graph, state, batch, jax.random.PRNGKey(4), critic_epochs=3, num_minibatches=2
    )
    actual, _, _, _ = nnx.merge(graph, actual_state)
    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7),
        nnx.state(expected),
        nnx.state(actual),
    )
    assert metrics.actor_steps == 1
    assert metrics.critic_steps == 6
    np.testing.assert_allclose(metrics.actor_loss_mean, loss)


def test_ppo_surrogate_both_advantage_signs_and_critic_continues():
    actor, critic, batch = fixture()
    ratio = jnp.array([0.5, 1.5, 0.5, 1.5])
    batch = batch._replace(logp=batch.logp - jnp.log(ratio))
    expected = -jnp.minimum(
        ratio * batch.advantages, jnp.clip(ratio, 0.8, 1.2) * batch.advantages
    ).mean()
    np.testing.assert_allclose(ppo_loss(actor, batch, 0.2), expected, atol=1e-6)
    graph, state = nnx.split(
        (
            actor,
            nnx.Optimizer(actor, optax.adam(0.1)),
            critic,
            nnx.Optimizer(critic, optax.adam(0.01)),
        )
    )
    _, metrics = update_ppo(
        graph,
        state,
        batch,
        jax.random.PRNGKey(4),
        actor_epochs=5,
        critic_epochs=3,
        num_minibatches=2,
        clip_ratio=0.2,
        target_kl=1e-12,
    )
    assert (
        metrics.actor_steps == 2
    )  # threshold checked over the rollout after each epoch
    assert metrics.critic_steps == 6


def test_gae_and_returns_match_scalar_reference_with_mixed_boundaries():
    rewards = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    values = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
    done = np.array([[False, True], [True, False], [False, True]])
    boot = np.array([[0.0, 0.0], [2.0, 0.0], [3.0, 4.0]], dtype=np.float32)
    expected_adv = np.zeros_like(values)
    expected_returns = np.zeros_like(values)
    for env in range(2):
        advantage = 0.0
        ret = next_value = boot[-1, env]
        for t in reversed(range(3)):
            if done[t, env]:
                next_value = ret = boot[t, env]
                advantage = 0.0
            advantage = (
                rewards[t, env]
                + 0.9 * next_value
                - values[t, env]
                + 0.9 * 0.7 * advantage
            )
            ret = rewards[t, env] + 0.9 * ret
            expected_adv[t, env] = advantage
            expected_returns[t, env] = ret
            next_value = values[t, env]
    advantages, returns = advantages_and_returns(
        jnp.array(rewards),
        jnp.array(values),
        jnp.array(done),
        jnp.array(boot),
        0.9,
        0.7,
    )
    np.testing.assert_allclose(advantages, expected_adv, atol=1e-6)
    np.testing.assert_allclose(returns, expected_returns, atol=1e-6)


def test_ppo_stopping_skips_later_epoch_inference(monkeypatch):
    actor, critic, batch = fixture()
    graph, state = nnx.split(
        (
            actor,
            nnx.Optimizer(actor, optax.adam(0.1)),
            critic,
            nnx.Optimizer(critic, optax.adam(0.01)),
        )
    )
    calls = []
    original = Actor.distribution_parameters

    def counted(model, observations):
        # Full-rollout calls are the initial reference, epoch KL and final diagnostics.
        if observations.shape[0] == batch.observations.shape[0]:
            jax.debug.callback(lambda: calls.append(1), ordered=True)
        return original(model, observations)

    update_ppo.clear_cache()
    monkeypatch.setattr(Actor, "distribution_parameters", counted)
    try:
        result, metrics = update_ppo(
            graph,
            state,
            batch,
            jax.random.PRNGKey(4),
            actor_epochs=5,
            critic_epochs=1,
            num_minibatches=2,
            clip_ratio=0.2,
            target_kl=1e-12,
        )
        jax.block_until_ready(result)
        jax.effects_barrier()
        assert int(metrics.actor_steps) == 2
        assert len(calls) == 4  # Additional stopped epochs perform no KL inference.
    finally:
        update_ppo.clear_cache()
