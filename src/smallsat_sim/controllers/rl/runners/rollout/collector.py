"""Build once; model states, simulation states and RNGs are runtime arguments."""

from flax import nnx
import jax
import jax.numpy as jnp

from .core import run_functional_rollout
from .types import FunctionalRolloutCallbacks, PolicyState, TransitionLabels
from .context import ContextHistory, append_history, reset_history, estimate_context


def make_collector(
    env,
    agent,
    am,
    *,
    steps,
    context_source="privileged",
    stochastic=True,
    demonstration=None,
    visualization=None,
):
    if context_source not in ("privileged", "estimated", "zero"):
        raise ValueError("Unknown context source")
    if env.use_adaptive_approach and context_source == "estimated" and am is None:
        raise ValueError("Estimated context requires an adaptation module")
    # Backend and model structure are fixed; their array state is a runtime input.
    config = env.build_step_config()
    backend = env.rollout_backend()
    features, physics_step, reset_fn = backend.state_features, backend.step, backend.reset
    max_episode_len = agent.max_ep_len if agent is not None else env.max_episode_len

    actor_graph = nnx.graphdef(agent.actor) if agent is not None else None
    critic_graph = nnx.graphdef(agent.critic) if agent is not None else None
    am_graph = nnx.graphdef(am) if am is not None else None
    scale = jnp.asarray(env.env_cfg.context.scale, dtype=jnp.float32)[: env.res_dim]
    adaptive = env.use_adaptive_approach

    @jax.jit
    def collect(state, actor_state, critic_state, am_state, key, reference):
        actor = nnx.merge(actor_graph, actor_state) if actor_graph is not None else None
        critic = (
            nnx.merge(critic_graph, critic_state) if critic_graph is not None else None
        )
        estimator = nnx.merge(am_graph, am_state) if am_graph is not None else None

        # Policy input contains the state and the context available before acting.
        def prepare_policy_input(_t, states, context, policy_state):
            return jnp.concatenate((states, context / scale), axis=-1), policy_state

        def sample_policy(_t, observations, key, policy_state):
            key, sample_key = jax.random.split(key)
            if demonstration is not None:
                actions = demonstration(observations[:, : env.obs_dim])
                latent, logp = jnp.zeros_like(actions), jnp.zeros((env.num_envs,))
            elif stochastic:
                actions, latent, logp = actor.sample(observations, sample_key)
            else:
                latent = actor.mu_net(observations)
                actions = actor.apply_action_bounds(latent)
                logp = actor.log_prob(observations, latent)
            values = (
                critic(observations)
                if critic is not None
                else jnp.zeros((env.num_envs,))
            )
            next_policy_state = PolicyState(
                history=policy_state.history, latent_actions=latent,
            )
            return actions, values, logp, key, next_policy_state

        # Label the history ending at this transition, then clear finished histories.
        def update_context(_t, output, actions, context, reset_mask, policy_state):
            if adaptive:
                history = append_history(policy_state.history, output.prev_states, actions)
                true_residual = output.actual_wrench - actions @ config.thruster_mixer_T
                if context_source == "estimated":
                    next_context = estimate_context(estimator, history, scale)
                elif context_source == "privileged":
                    next_context = true_residual
                else:
                    next_context = jnp.zeros_like(context)
                history_full = history.counts >= env.history_len
                policy_state = PolicyState(
                    history=reset_history(history, reset_mask),
                    latent_actions=policy_state.latent_actions,
                )
            else:
                next_context, true_residual = context, context
                history_full = jnp.zeros((env.num_envs,), dtype=bool)
            return (
                next_context,
                TransitionLabels(
                    latent_actions=policy_state.latent_actions,
                    normalized_context=true_residual / scale,
                    history_full=history_full,
                ),
                policy_state,
            )

        # Bootstrap never samples an action or advances estimator history.
        def bootstrap_value(_t, state, context, key, policy_state):
            observations, _ = prepare_policy_input(_t, features(state, reference), context, policy_state)
            values = (
                critic(observations)
                if critic is not None
                else jnp.zeros((env.num_envs,))
            )
            return values, key, policy_state

        history = ContextHistory(
            values=jnp.zeros(
                (
                    env.num_envs,
                    env.history_len if adaptive else 0,
                    env.obs_dim + env.act_dim if adaptive else 0,
                )
            ),
            counts=jnp.zeros((env.num_envs,), dtype=jnp.int32),
        )
        return run_functional_rollout(
            step_config=config,
            max_episode_len=max_episode_len,
            initial_state=state,
            initial_context=jnp.zeros((env.num_envs, env.res_dim)),
            rng=key,
            num_steps=steps,
            reference_waypoint=reference,
            callbacks=FunctionalRolloutCallbacks(
                prepare_policy_input=prepare_policy_input,
                sample_policy=sample_policy,
                update_context=update_context,
                bootstrap_value=bootstrap_value,
            ),
            policy_state=PolicyState(
                history=history,
                latent_actions=jnp.zeros((env.num_envs, env.act_dim)),
            ),
            state_features_fn=features,
            step_fn=physics_step,
            reset_fn=reset_fn,
            visualization=visualization,
        )

    return collect
