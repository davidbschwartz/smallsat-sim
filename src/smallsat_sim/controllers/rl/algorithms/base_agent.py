"""Common model ownership for PPO and VPG; algorithm differences stay explicit."""

from flax import nnx
import jax
import jax.numpy as jnp
import optax

from ..modules.base_network import Critic
from ..modules.base_policy import Actor
from ..storage.rollout_batch import normalize_advantages
from smallsat_sim.controllers.base_controller import BaseController

DEFAULT_POLICY_HIDDEN_SIZES = (64, 64)
DEFAULT_INITIAL_LOG_STD = -0.5


def policy_hidden_sizes_from_config(cfg):
    return tuple(getattr(cfg, "policy_hidden_sizes", DEFAULT_POLICY_HIDDEN_SIZES))


class BaseAgent(BaseController):
    algorithm = None

    def __init__(self, env, planner, rng_key, activation=nnx.tanh, *, config):
        cfg = config
        super().__init__(env, planner, env.env_cfg.environment)
        self.env, self.planner, self.ctrl_cfg = env, planner, cfg
        hp = getattr(cfg, self.algorithm.upper())
        settings = (
            "steps_per_epoch",
            "epochs",
            "max_ep_len",
            "gamma",
            "lam",
            "actor_lr",
            "critic_lr",
            "critic_training_epochs",
        )
        for name in settings:
            setattr(self, name, getattr(hp, name))
        self.num_minibatches = int(getattr(hp, "num_minibatches", 1))
        self.batch_size = env.num_envs * self.steps_per_epoch
        if self.num_minibatches < 1 or self.batch_size % self.num_minibatches:
            raise ValueError("Rollout size must be divisible by num_minibatches")
        if (
            min(
                self.steps_per_epoch,
                self.epochs,
                self.max_ep_len,
                self.critic_training_epochs,
            )
            <= 0
        ):
            raise ValueError("Training horizons and update counts must be positive")
        self.key, model_key = jax.random.split(rng_key)
        self.key = jax.device_put(self.key, self.key.sharding)
        rngs = nnx.Rngs(model_key)
        ranges = jnp.array(
            [t.forcerange for t in env.model_cfg.actuators],
            dtype=jnp.float32,
        )
        sizes = policy_hidden_sizes_from_config(cfg)
        self.actor = Actor(
            env.obs_dim,
            env.act_dim,
            sizes,
            activation,
            env.res_dim,
            ranges[:, 0],
            ranges[:, 1],
            initial_log_std=getattr(
                hp,
                "initial_log_std",
                DEFAULT_INITIAL_LOG_STD,
            ),
            log_std_min=getattr(hp, "log_std_min", None),
            rngs=rngs,
        )
        self.critic = Critic(env.obs_dim, sizes, activation, env.res_dim, rngs=rngs)
        self.reset_optimizers()

    def reset_optimizers(self):
        self.actor_optimizer = nnx.Optimizer(
            self.actor, optax.adam(self.actor_lr, eps=1e-5)
        )
        self.critic_optimizer = nnx.Optimizer(
            self.critic,
            optax.adam(self.critic_lr, eps=1e-5),
        )
        # Commit initial state to the same device as compiled rollout outputs.
        objects = (self.actor, self.actor_optimizer, self.critic, self.critic_optimizer)
        nnx.update(objects, jax.device_put(nnx.state(objects), self.key.sharding))

    def update(self, batch):
        batch = batch._replace(advantages=normalize_advantages(batch.advantages))
        self.key, key = jax.random.split(self.key)
        objects = (self.actor, self.actor_optimizer, self.critic, self.critic_optimizer)
        graph, state = nnx.split(objects)
        state, metrics = self.update_parameters(graph, state, batch, key)
        nnx.update(objects, state)
        return metrics

    def act(self, observations, log=False):
        self.key, key = jax.random.split(self.key)
        sample = self.actor.sample(observations, key)
        return sample.actions, self.critic(observations), sample.logp

    def get_control_input(self, stage=None, obs_residuals=None):
        if obs_residuals is None:
            raise ValueError("obs_residuals is required")
        return self.actor.deterministic_action(obs_residuals)

    def _log(self, run_id, timestamp, env):
        """The runner reports aggregated metrics after each rollout."""
