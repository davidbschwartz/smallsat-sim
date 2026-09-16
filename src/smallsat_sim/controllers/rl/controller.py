"""Continuous path tracking using the same history/context convention as training."""

from pathlib import Path
from types import SimpleNamespace
from .algorithms.sac import make_actor

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from .config import validate_policy_context
from .algorithms.ppo import PPO
from .algorithms.vpg import VPG
from .modules import adaptation_model
from .runners.rollout.context import ContextHistory, append_history, estimate_context
from .runners.runner_setup import build_checkpoint_file_names
from .runners.runner_utils import (
    restore_trained_modules,
    load_trained_modules,
    update_module_from_checkpoint_state,
    model_fingerprint,
    resolve_checkpoint_paths,
)
from smallsat_sim.controllers.pd.vectorized_controller import VectorizedPDController
from smallsat_sim.envs.vec_env.freeflyer import to_mjx


@nnx.jit
def policy_action(actor, observations):
    return actor.deterministic_action(observations)


@nnx.jit
def predict_context(model, history, scale):
    return estimate_context(model, history, scale)


class RLController:
    def __init__(
        self, env, planner, ckpt_name=None, *, config, test_pd=False, checkpoint=None,
    ):
        validate_policy_context(env.env_cfg, config)
        self.env, self.planner = env, planner
        self.training_config = config
        self.checkpoint = Path(checkpoint) if checkpoint is not None else None
        cfg = config
        self._rng, key = jax.random.split(env.next_rng_keys(1)[0])
        self.agent = (
            None
            if test_pd
            else SimpleNamespace(actor=make_actor(env, config, key)) if cfg.algorithm == 'sac'
            else {"ppo": PPO, "vpg": VPG}[cfg.algorithm](
                env,
                planner,
                key,
                config=config,
            )
        )
        self.am = (
            None
            if test_pd
            else adaptation_model(
                cfg.am_architecture if env.use_adaptive_approach else None,
                history_length=env.history_len,
                input_size=env.obs_dim + env.act_dim,
                context_size=env.res_dim,
                key=self._take_keys(),
            )
        )
        self.pd_ctrl = VectorizedPDController(env, planner) if test_pd else None
        self.deployment_len = cfg.deployment_len
        self.ckpt_dir = cfg.checkpoint_dir
        names = build_checkpoint_file_names(env, cfg.algorithm)
        self.ckpt_filename = ckpt_name or names["training_state_file_name"]
        if cfg.algorithm == 'sac' and ckpt_name is None:
            self.ckpt_filename += '_actor'
        self.adaptation_module_file_name = names["adaptation_module_file_name"]
        self.context_scale = jnp.asarray(cfg.context_scale)[: env.res_dim]

    def _take_keys(self):
        self._rng, key = jax.random.split(self._rng)
        return key

    def _validate_policy_checkpoint(self, payload):
        env = self.env
        saved_config = payload["metadata"]["resolved_config"]
        saved = saved_config["rl"]
        # A custom training asset must also be supplied to the deployment env.
        for name, value in saved_config.get("physical_properties", {}).items():
            if not np.array_equal(value, getattr(env.model_cfg.physical, name)):
                raise ValueError(f"Deployment physical property differs at {name}")
        if "thruster_mixer" in saved_config and not np.array_equal(
            saved_config["thruster_mixer"], env._thruster_mixer_T
        ):
            raise ValueError(
                "Deployment thruster geometry differs from the checkpoint"
            )
        cfg = self.training_config
        for name in (
            "algorithm",
            "use_adaptive_approach",
            "am_architecture",
            "context_window_len",
            "context_scale",
            "policy_hidden_sizes",
        ):
            if name in saved and np.any(
                np.asarray(saved[name]) != np.asarray(getattr(cfg, name))
            ):
                raise ValueError(f"Deployment configuration differs at {name}")
        for name in ("act_low", "act_high"):
            if not np.array_equal(
                payload["actor_model"][name],
                getattr(self.agent.actor, name),
            ):
                raise ValueError(
                    "Deployment actuator bounds differ from the checkpoint"
                )

    def _load_policy(self):
        policy_path = Path(self.ckpt_dir) / self.ckpt_filename
        adaptation_path = Path(self.ckpt_dir) / self.adaptation_module_file_name
        if self.checkpoint is not None:
            supplied = load_trained_modules(self.checkpoint.parent, self.checkpoint.name)
            policy_path, adaptation_path = resolve_checkpoint_paths(
                self.checkpoint, stage=supplied["metadata"]["stage"],
                policy_filename=self.ckpt_filename,
                adaptation_filename=self.adaptation_module_file_name,
            )
        payload = load_trained_modules(policy_path.parent, policy_path.name)
        self._validate_policy_checkpoint(payload)
        if self.training_config.algorithm == 'sac':
            update_module_from_checkpoint_state(self.agent.actor, payload['actor_model'])
        else:
            restore_trained_modules(self.agent, policy_path.parent, policy_path.name)
        if self.am is not None:
            state = load_trained_modules(
                adaptation_path.parent, adaptation_path.name
            )
            if state["metadata"]["teacher_id"] != model_fingerprint(
                self.agent.actor
            ):
                raise ValueError(
                    "Adaptation checkpoint belongs to a different teacher policy"
                )
            update_module_from_checkpoint_state(self.am, state["am_model"])

    def _prepare_deployment(self):
        env = self.env
        # Deployment tracks continuously; training success flags do not reset it.
        env.reset()
        env.reset_perturbations()
        env.reset_disturbances()
        backend = env.env_cfg.environment.rollout_backend
        step_config = env.build_step_config()
        transition = env.rollout_backend().step_with_observations
        # Configuration is fixed for this deployment; effects are dynamic state.
        step_fn = jax.jit(
            lambda state, actions, reference, context:
                transition(state, actions, reference, step_config, context)
        )
        state = (
            env.freeflyer_state_struct() if backend == "freeflyer" else env.state_struct
        )
        return backend, step_config, step_fn, state

    def _log_step(self, step, stage, states, output):
        env = self.env
        if hasattr(env, "logger"):
            env.logger.log(
                env.run_id,
                float(step),
                run_name=env.run_name,
                stage=stage,
                mean_lateral_error=float(
                    jnp.linalg.norm(states[:, :3], axis=-1).mean()
                ),
                mean_angle_error=float(
                    jnp.rad2deg(jnp.linalg.norm(states[:, 3:6], axis=-1)).mean()
                ),
                success_terminal_rate=float(output.success_terminals.mean()),
                failure_terminal_rate=float(output.failure_terminals.mean()),
            )

    def control(
        self,
        stage="deployment",
        phase=2,
        test_pd=False,
        perturbation_distribution=None,
        perturbation_distributions=None,
        apply_disturbances=False,
        perturbation_sequence=None,
        disturbance_start_steps=(),
    ):
        env = self.env
        if env.use_adaptive_approach and phase != 2:
            raise ValueError("Deployment uses estimated context only (phase=2)")
        if test_pd:
            if self.pd_ctrl is None:
                self.pd_ctrl = VectorizedPDController(env, self.planner)
        else:
            self._load_policy()
        backend, step_config, step_fn, state = self._prepare_deployment()
        # Each estimate uses only completed state/action pairs.
        history = ContextHistory(
            jnp.zeros((env.num_envs, env.history_len, env.obs_dim + env.act_dim)),
            jnp.zeros((env.num_envs,), dtype=jnp.int32),
        )
        context = jnp.zeros((env.num_envs, env.res_dim))
        reference = self.planner.get_reference(env.get_obs())
        states = env.get_states(reference)
        events = list(perturbation_sequence or ())
        disturbance_events = list(disturbance_start_steps)
        if perturbation_sequence is None:
            distributions = perturbation_distributions or (
                ()
                if perturbation_distribution is None
                else (perturbation_distribution,)
            )
            events = [(100, distribution) for distribution in distributions]
            if apply_disturbances:
                disturbance_events.append(100)
        visualization = env.rollout_visualization("evaluation", step_config)
        step = 0
        while self.deployment_len is None or step < self.deployment_len:
            for onset, distribution in events:
                if step == onset:
                    env.schedule_random_faults(self._take_keys(), 1.0, distribution)
            for onset in disturbance_events:
                if step == onset:
                    env.schedule_random_disturbances(self._take_keys(), 1.0)
            state = state.replace(
                perturbation_states=env.perturbation_states,
                disturbance_states=env.disturbance_states,
            )
            if test_pd:
                actions = self.pd_ctrl.get_control_input(env, reference)
            else:
                observations = jnp.concatenate(
                    (states, context / self.context_scale), axis=-1
                )
                actions = policy_action(self.agent.actor, observations)
            # Advance physics, then prepare context for the next policy action.
            state, output = step_fn(state, actions, reference, context)
            if self.am is not None:
                history = append_history(history, output.prev_states, actions)
                context = predict_context(self.am, history, self.context_scale)
            synced = (
                to_mjx(
                    state,
                    env.state_struct,
                    step_config,
                )
                if backend == "freeflyer"
                else state
            )
            env.apply_state_struct(synced)
            reference = self.planner.get_reference(output.next_obs)
            states = env.get_states(reference)
            self._log_step(step, stage, states, output)
            if visualization is not None:
                visualization.observe(
                    jnp.asarray(step),
                    state,
                    jnp.zeros(env.num_envs, dtype=bool),
                )
            if bool(self.planner.completed_path.all()):
                break
            step += 1

        if visualization is not None:
            jax.effects_barrier()
            visualization.finish()
