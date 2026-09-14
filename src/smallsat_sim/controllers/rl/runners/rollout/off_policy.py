"""Persistent SAC collection; replay always sees observations before masked reset."""
from typing import NamedTuple
from flax import nnx
import jax
import jax.numpy as jnp
from ...storage.replay_buffer import Transition
from .off_policy_effects import merge_effects


class OffPolicyCarry(NamedTuple):
    state: object
    features: jax.Array
    episode_returns: jax.Array
    episode_lengths: jax.Array
    key: jax.Array


class OffPolicyResult(NamedTuple):
    transitions: Transition
    step_outputs: object
    actions: jax.Array
    episode_returns: jax.Array
    done_masks: jax.Array
    truncated_masks: jax.Array


def initial_carry(state, backend, reference, key, num_envs):
    return OffPolicyCarry(state, backend.state_features(state, reference),
                          jnp.zeros(num_envs), jnp.zeros(num_envs, jnp.int32), key)


def make_off_policy_collector(env, actor, *, steps, max_ep_len, stochastic=True,
                              visualization=None):
    backend, config = env.rollout_backend(), env.build_step_config()
    graph = nnx.graphdef(actor)
    num_envs = env.num_envs
    # Explicit environment IDs are part of the task, not randomizable labels.
    preserve_env_ids = any(spec.env_ids is not None for spec in getattr(env, 'fault_specs', ()))

    @jax.jit
    def collect(carry, actor_state, reference, collected, warmup, effect_pool=None):
        policy = nnx.merge(graph, actor_state)

        def step(carry, index):
            key, action_key = jax.random.split(carry.key)
            if stochastic:
                actions = jax.lax.cond(
                    collected + index * num_envs < warmup,
                    lambda k: jax.random.uniform(k, (num_envs, env.act_dim),
                                                minval=policy.act_low, maxval=policy.act_high),
                    lambda k: policy.sample(carry.features, k).actions, action_key)
            else:
                actions = policy.deterministic_action(carry.features)
            state, output, next_features = backend.step(
                carry.state, actions, reference, config,
                jnp.zeros((num_envs, 0)), carry.features)
            lengths = carry.episode_lengths + 1
            returns = carry.episode_returns + output.rewards
            terminated = output.terminals.astype(bool)
            truncated = (lengths >= max_ep_len) & ~terminated
            done = terminated | truncated
            if visualization is not None:
                visualization.observe(index, state, done)
            transition = Transition(carry.features, actions, output.rewards,
                                    next_features, terminated, truncated)
            if effect_pool is not None:
                key, reset_key = jax.random.split(key)
                # All non-RNG leaves have a leading pool axis.
                pool_size = effect_pool[0][0].active_mask.shape[0]
                if preserve_env_ids:
                    candidates = jax.random.randint(reset_key, (num_envs,), 0, pool_size // num_envs)
                    rows = candidates * num_envs + jnp.arange(num_envs)
                else:
                    rows = jax.random.randint(reset_key, (num_envs,), 0, pool_size)
                disturbances, faults = jax.tree_util.tree_map_with_path(
                    lambda path, x: x if getattr(path[-1], 'name', None) == 'rng' else x[rows],
                    effect_pool)
                state = state.replace(
                    disturbance_states=merge_effects(disturbances, state.disturbance_states, done),
                    perturbation_states=merge_effects(faults, state.perturbation_states, done),
                )
            state = jax.lax.cond(jnp.any(done),
                                 lambda s: backend.reset(s, config, done), lambda s: s, state)
            features = jnp.where(done[:, None], backend.state_features(state, reference), next_features)
            carry = OffPolicyCarry(state, features, jnp.where(done, 0., returns),
                                   jnp.where(done, 0, lengths), key)
            result = OffPolicyResult(transition, output, actions,
                                     jnp.where(done, returns, 0.), done, truncated)
            return carry, result
        return jax.lax.scan(step, carry, jnp.arange(steps))
    return collect
