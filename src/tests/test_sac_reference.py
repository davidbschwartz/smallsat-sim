"""Independent SAC audit against paper equations and Spinning Up update order.

The Torch reference uses SGD so every parameter delta directly checks gradients,
including actor backpropagation through Q, frozen targets, and temperature sign.
"""
from types import SimpleNamespace

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from smallsat_sim.controllers.rl.algorithms.sac import SAC, update_sac
from smallsat_sim.controllers.rl.config import load_training_config
from smallsat_sim.controllers.rl.storage.replay_buffer import Transition


def test_full_sac_step_matches_independent_torch_reference():
    torch = pytest.importorskip('torch')
    functional = torch.nn.functional
    config = load_training_config()
    config.policy_hidden_sizes = [4, 4]
    env = SimpleNamespace(obs_dim=3, act_dim=2, model_cfg=SimpleNamespace(actuators=[
        SimpleNamespace(forcerange=(-2., 4.)), SimpleNamespace(forcerange=(3., 5.))]))
    agent = SAC(env, None, jax.random.PRNGKey(71), config=config)
    # Different target values detect accidentally bootstrapping with online Qs.
    nnx.update(agent.targets, jax.tree.map(lambda x: x + .13, nnx.state(agent.targets)))
    lr = .01
    agent.actor_optimizer = nnx.Optimizer(agent.actor, optax.sgd(lr))
    agent.critic_optimizer = nnx.Optimizer(agent.critics, optax.sgd(lr))
    agent.alpha_optimizer = nnx.Optimizer(agent.temperature, optax.sgd(lr))

    def tensor(x, grad=False):
        return torch.tensor(np.asarray(x).copy(), dtype=torch.float32, requires_grad=grad)

    def network(model, grad=True):
        return [(tensor(layer.kernel.value, grad), tensor(layer.bias.value, grad))
                for layer in model.layers if isinstance(layer, nnx.Linear)]

    def forward(layers, x):
        for index, (weight, bias) in enumerate(layers):
            x = x @ weight + bias
            if index < len(layers) - 1:
                x = functional.relu(x)
        return x

    actor = network(agent.actor.net)
    critics = [network(agent.critics.q1), network(agent.critics.q2)]
    targets = [network(agent.targets.q1, False), network(agent.targets.q2, False)]
    log_alpha = tensor(agent.temperature.log_alpha.value, True)
    alpha = log_alpha.detach().exp()
    actor_parameters = [p for layer in actor for p in layer]
    critic_parameters = [p for layers in critics for layer in layers for p in layer]
    actor_opt = torch.optim.SGD(actor_parameters, lr=lr)
    critic_opt = torch.optim.SGD(critic_parameters, lr=lr)
    alpha_opt = torch.optim.SGD([log_alpha], lr=lr)
    key = jax.random.PRNGKey(37)
    next_key, actor_key = jax.random.split(key)
    obs = jnp.array([[.1, -.3, 1.], [.5, .2, -.7], [-.4, .8, .3]])
    next_obs = obs * .7 + .2
    actions = jnp.array([[-1., 4.5], [2., 3.3], [.4, 4.1]])
    batch = Transition(obs, actions, jnp.array([.2, -.7, 1.1]), next_obs,
                       jnp.array([True, False, False]), jnp.array([False, True, False]))

    def policy(observations, noise_key):
        mean, log_std = forward(actor, observations).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, config.SAC.log_std_min, config.SAC.log_std_max)
        noise = tensor(jax.random.normal(noise_key, mean.shape))
        latent = mean + log_std.exp() * noise
        # Independent change of variables using the transformed distribution.
        normal = torch.distributions.Normal(mean, log_std.exp())
        logp = normal.log_prob(latent) - functional.logsigmoid(latent) - functional.logsigmoid(-latent)
        return torch.sigmoid(latent), logp.sum(dim=-1)

    def q_values(layers, observations, normalized_actions):
        inputs = torch.cat([observations, normalized_actions], dim=-1)
        return [forward(model, inputs).squeeze(-1) for model in layers]

    with torch.no_grad():
        next_action, next_logp = policy(tensor(next_obs), next_key)
        q1, q2 = q_values(targets, tensor(next_obs), next_action)
        target = tensor(batch.rewards) + agent.gamma * (1 - tensor(batch.terminated)) * (
            torch.minimum(q1, q2) - alpha * next_logp)
    normalized = (tensor(actions) - tensor(agent.actor.act_low)) / tensor(agent.actor.act_range)
    q1, q2 = q_values(critics, tensor(obs), normalized)
    critic_loss = ((q1 - target).square() + (q2 - target).square()).mean()
    critic_loss.backward()
    critic_opt.step()

    for parameter in critic_parameters:
        parameter.requires_grad_(False)
    action, logp = policy(tensor(obs), actor_key)
    q1, q2 = q_values(critics, tensor(obs), action)
    actor_loss = (alpha * logp - torch.minimum(q1, q2)).mean()
    actor_loss.backward()
    actor_opt.step()
    alpha_loss = -log_alpha * (logp.detach().mean() + agent.target_entropy)
    alpha_loss.backward()
    alpha_opt.step()
    with torch.no_grad():
        for target_layers, online_layers in zip(targets, critics):
            for target_pair, online_pair in zip(target_layers, online_layers):
                for target_parameter, online_parameter in zip(target_pair, online_pair):
                    target_parameter.mul_(1 - agent.tau).add_(agent.tau * online_parameter)

    metrics = update_sac(*agent.objects(), batch, key, agent.gamma, agent.tau, agent.target_entropy)
    for actual, expected in ((metrics['critic_loss_mean'], critic_loss),
                             (metrics['actor_loss_mean'], actor_loss),
                             (metrics['alpha_loss'], alpha_loss),
                             (agent.temperature.log_alpha.value, log_alpha)):
        np.testing.assert_allclose(actual, expected.detach().numpy(), rtol=2e-5, atol=2e-6)
    for actual, expected in ((agent.actor.net, actor), (agent.critics.q1, critics[0]),
                             (agent.critics.q2, critics[1]), (agent.targets.q1, targets[0]),
                             (agent.targets.q2, targets[1])):
        for actual_pair, expected_pair in zip(network(actual, False), expected):
            for actual_parameter, expected_parameter in zip(actual_pair, expected_pair):
                np.testing.assert_allclose(actual_parameter.numpy(), expected_parameter.detach().numpy(),
                                           rtol=2e-5, atol=2e-6)
