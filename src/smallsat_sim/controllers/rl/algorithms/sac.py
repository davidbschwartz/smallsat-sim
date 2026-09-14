"""Modern SAC with twin Q targets and automatic normalized-action entropy tuning."""
from flax import nnx
import jax
import jax.numpy as jnp
import optax

from ..modules.sac_policy import SACActor
from ..modules.q_network import TwinQ


def make_actor(env, config, key):
    ranges = jnp.asarray([t.forcerange for t in env.model_cfg.actuators], jnp.float32)
    return SACActor(env.obs_dim, ranges[:, 0], ranges[:, 1],
                    config.policy_hidden_sizes, rngs=nnx.Rngs(key),
                    log_std_min=config.SAC.log_std_min, log_std_max=config.SAC.log_std_max)


class Temperature(nnx.Module):
    def __init__(self, initial):
        self.log_alpha = nnx.Param(jnp.log(jnp.asarray(initial, jnp.float32)))


def bellman_target(rewards, terminated, next_q, logp, gamma, alpha):
    return jax.lax.stop_gradient(rewards + gamma * (1 - terminated) * (next_q - alpha * logp))


class SAC:
    algorithm = 'sac'

    def __init__(self, env, planner, key, *, config):
        self.has_logger = hasattr(env, 'logger')
        self.key, actor_key, critic_key = jax.random.split(key, 3)
        self.actor = make_actor(env, config, actor_key)
        self.critics = TwinQ(env.obs_dim, env.act_dim, config.policy_hidden_sizes,
                            rngs=nnx.Rngs(critic_key))
        self.targets = nnx.clone(self.critics)
        hp = config.SAC
        self.temperature = Temperature(hp.initial_alpha)
        self.actor_optimizer = nnx.Optimizer(self.actor, optax.adam(hp.actor_lr))
        self.critic_optimizer = nnx.Optimizer(self.critics, optax.adam(hp.critic_lr))
        self.alpha_optimizer = nnx.Optimizer(self.temperature, optax.adam(hp.alpha_lr))
        self.gamma, self.tau = hp.gamma, hp.tau
        self.target_entropy = (-float(env.act_dim) if hp.target_entropy is None
                               else float(hp.target_entropy))
        self.max_ep_len = hp.max_ep_len
        self.steps_per_epoch = hp.collection_steps

    def objects(self):
        return (self.actor, self.critics, self.targets, self.temperature,
                self.actor_optimizer, self.critic_optimizer, self.alpha_optimizer)

    def update(self, batch):
        self.key, key = jax.random.split(self.key)
        return update_sac(*self.objects(), batch, key, self.gamma, self.tau, self.target_entropy)


@nnx.jit
def update_sac(actor, critics, targets, temperature, actor_optimizer,
               critic_optimizer, alpha_optimizer, batch, key, gamma, tau, target_entropy):
    next_key, actor_key = jax.random.split(key)
    alpha = jax.lax.stop_gradient(jnp.exp(temperature.log_alpha.value))
    next_sample = actor.sample(batch.next_observations, next_key)
    q1, q2 = targets(batch.next_observations, actor.normalize_action(next_sample.actions))
    target = bellman_target(batch.rewards, batch.terminated, jnp.minimum(q1, q2),
                            next_sample.logp, gamma, alpha)

    def critic_loss(model):
        q1, q2 = model(batch.observations, actor.normalize_action(batch.actions))
        return ((q1 - target)**2 + (q2 - target)**2).mean()
    critic_loss_value, grads = nnx.value_and_grad(critic_loss)(critics)
    critic_optimizer.update(grads)

    def actor_loss(model):
        draw = model.sample(batch.observations, actor_key)
        q1, q2 = critics(batch.observations, model.normalize_action(draw.actions))
        return (alpha * draw.logp - jnp.minimum(q1, q2)).mean(), draw.logp.mean()
    (actor_loss_value, logp), grads = nnx.value_and_grad(actor_loss, has_aux=True)(actor)
    actor_optimizer.update(grads)
    # Log-temperature surrogate: the dual's stationary entropy condition, with
    # its gradient rescaled by 1 / alpha relative to Eq. 18 in log coordinates.
    entropy_error = jax.lax.stop_gradient(logp + target_entropy)
    alpha_loss, grads = nnx.value_and_grad(
        lambda model: -model.log_alpha.value * entropy_error)(temperature)
    alpha_optimizer.update(grads)
    nnx.update(targets, jax.tree.map(lambda old, new: (1 - tau) * old + tau * new,
                                   nnx.state(targets), nnx.state(critics)))
    return {'actor_loss_mean': actor_loss_value, 'critic_loss_mean': critic_loss_value,
            'alpha_loss': alpha_loss, 'alpha': jnp.exp(temperature.log_alpha.value),
            'entropy': -logp, 'target_q_mean': target.mean()}
