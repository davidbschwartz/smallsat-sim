"""Own training state and I/O; numerical work lives in compiled functions."""

from smallsat_sim.envs.effects.catalog import fault_weights

from copy import deepcopy
from pathlib import Path
import hashlib
import json

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb

from ..config import validate_policy_context, config_dict
from ..algorithms.ppo import PPO
from ..algorithms.vpg import VPG
from ..modules import adaptation_model
from .adaptation_training import (
    train_adaptation_module_on_policy_runner,
    initialize_from_teacher,
)
from .evaluation_loop import evaluate_runner
from .pretraining import pretrain
from .rollout.collector import make_collector
from .runner_setup import build_checkpoint_file_names, configure_jax_compilation_cache
from .runner_utils import (
    checkpoint_exists,
    load_trained_modules,
    save_trained_modules,
    restore_trained_modules,
    save_adaptation_module,
    model_fingerprint,
    update_module_from_checkpoint_state,
    _restore_optimizer,
    _checked_state,
)
from .training_loop import learn_runner
from smallsat_sim.envs.effects.scheduling import reset_and_randomize




class OnPolicyRunner:
    def __init__(self, env, planner, *, config):
        validate_policy_context(env.env_cfg, config)
        configure_jax_compilation_cache(jax)
        self.env, self.planner = env, planner
        self.training_cfg = deepcopy(config)

        # Validate the learning method before allocating its models.
        algorithm = self.training_cfg.algorithm.lower()
        if algorithm not in ("ppo", "vpg"):
            raise ValueError("algorithm must be ppo or vpg")
        if algorithm == "vpg" and env.use_adaptive_approach:
            raise ValueError("The retained adaptation pipeline uses PPO")

        # Independent streams initialize the policy and optional history estimator.
        self._rng, key = jax.random.split(env.next_rng_keys(1)[0])
        self.agent = {"ppo": PPO, "vpg": VPG}[algorithm](env, planner, key, config=self.training_cfg)
        self.am = adaptation_model(
            self.training_cfg.am_architecture if env.use_adaptive_approach else None,
            history_length=env.history_len,
            input_size=env.obs_dim + env.act_dim,
            context_size=env.res_dim,
            key=self._take_keys(),
        )
        self.am_optimizer = (
            nnx.Optimizer(self.am, optax.adam(self.training_cfg.am_lr))
            if self.am is not None
            else None
        )
        if self.am is not None:
            objects = (self.am, self.am_optimizer)
            nnx.update(
                objects, jax.device_put(nnx.state(objects), self.agent.key.sharding)
            )

        # Context is expressed in physical units outside the neural networks.
        self.context_scale = jnp.asarray(
            self.training_cfg.context_scale,
            dtype=jnp.float32,
        )[: env.res_dim]
        if len(self.training_cfg.context_scale) != 6 or not np.all(
            np.asarray(self.training_cfg.context_scale) > 0
        ):
            raise ValueError(
                "context_scale must contain six positive force/torque scales"
            )

        # Resolve filenames and configuration once for strict checkpoint validation.
        self.reference_point = planner.get_reference(env.get_obs())
        self.ckpt_dir = str(Path(self.training_cfg.checkpoint_dir))
        for name, value in build_checkpoint_file_names(env, algorithm).items():
            setattr(self, name, value)
        self.policy_epoch = self.adaptation_epoch = 0
        self.teacher_source = None
        self.evaluation_checkpoint = None
        self._collectors = {}
        self._collector_modes = {}
        self.resolved_config = {
            # Keep the checkpoint schema stable while runtime settings have two owners.
            "rl": {
                **{
                    name: (fault_weights(value).tolist() if name == "failure_distribution" else value)
                    for name, value in config_dict(env.env_cfg.environment).items()
                    if not (name == "termination" and value == "full_pose")
                    and not (name in ("faults", "custom_faults", "custom_disturbances") and value == [])
                },
                # Adding off-policy defaults must not invalidate existing PPO/VPG runs.
                **{name: value for name, value in config_dict(self.training_cfg).items()
                   if name != "SAC"},
            },
            "bodies": config_dict(env.env_cfg.Bodies),
            "seed": int(env.env_cfg.sim.seed),
            "model": str(env.env_cfg.model),
            "physics_dt": float(env.model.opt.timestep),
            "physical_properties": config_dict(env.model_cfg.physical),
            "thruster_mixer": np.asarray(env._thruster_mixer_T).tolist(),
            "actuator_ranges": np.asarray(self.agent.actor.act_range).tolist(),
            "reference": np.asarray(self.reference_point).tolist(),
        }
        identity = json.dumps(self.resolved_config, sort_keys=True)
        self.config_id = hashlib.sha256(identity.encode()).hexdigest()
        if env.use_wandb:
            wandb.init(
                project="Astrobee-training",
                name=env.run_name,
                config=self.resolved_config,
            )

    def _take_keys(self, count=1):
        if count < 1:
            raise ValueError("count must be positive")
        keys = jax.random.split(self._rng, count + 1)
        self._rng = keys[0]
        return keys[1] if count == 1 else keys[1:]

    def collector(
        self,
        source,
        *,
        stochastic,
        steps=None,
        demonstration=None,
        mode="training",
    ):
        steps = steps or self.agent.steps_per_epoch
        cache_key = (source, stochastic, steps, demonstration, mode)
        if cache_key not in self._collectors:
            visualization = getattr(self.env, "_rl_visualization", None)
            self._collectors[cache_key] = make_collector(
                self.env,
                self.agent,
                self.am,
                steps=steps,
                context_source=source,
                stochastic=stochastic,
                demonstration=demonstration,
                visualization=visualization,
            )
            self._collector_modes[self._collectors[cache_key]] = mode
        return self._collectors[cache_key]

    def collect(self, collector, *, randomize):
        visualization = self.env.rollout_visualization(
            self._collector_modes[collector],
            self.env.build_step_config(),
        )
        reset_and_randomize(self.env, self._take_keys(), enabled=randomize)
        state = (
            self.env.freeflyer_state_struct()
            if self.env.env_cfg.environment.rollout_backend == "freeflyer"
            else self.env.state_struct
        )
        # Reset-created arrays and compiled outputs must have identical placement.
        state = jax.device_put(state, self.agent.key.sharding)
        result = collector(
            state,
            nnx.state(self.agent.actor),
            nnx.state(self.agent.critic),
            nnx.state(self.am) if self.am is not None else None,
            self.agent.key,
            self.reference_point,
        )
        if visualization is not None:
            jax.block_until_ready(result.actions)
            jax.effects_barrier()
            visualization.finish()
        self.agent.key = result.final_rng
        # The next collection starts a fresh horizon; only simulation RNG persists.
        self.env._rng = result.final_state.rng
        return result

    def restore_for_evaluation(self, source):
        """Load inference weights without rewinding optimizers, counters, or RNGs."""
        if source not in ("zero", "privileged", "estimated"):
            raise ValueError("Unknown context source")
        path = (Path(self.evaluation_checkpoint) if self.evaluation_checkpoint is not None
                else Path(self.ckpt_dir) / self._filename("policy"))
        policy = load_trained_modules(path.parent, path.name)
        if policy["metadata"]["config_id"] != self.config_id:
            raise ValueError("Checkpoint configuration differs from this run")
        actor_state = _checked_state(nnx.state(self.agent.actor), policy["actor_model"])
        if source == "estimated":
            if self.am is None:
                raise ValueError("Estimated evaluation requires an adaptation module")
            adaptation = load_trained_modules(path.parent, self._filename("adaptation"))
            if adaptation["metadata"]["config_id"] != self.config_id:
                raise ValueError("Checkpoint configuration differs from this run")
            teacher = nnx.merge(nnx.graphdef(self.agent.actor), actor_state)
            if adaptation["metadata"]["teacher_id"] != model_fingerprint(teacher):
                raise ValueError("Adaptation checkpoint belongs to a different teacher policy")
            am_state = _checked_state(nnx.state(self.am), adaptation["am_model"])
        nnx.update(self.agent.actor, actor_state)
        if source == "estimated":
            nnx.update(self.am, am_state)

    def collect_state(self, collector, state, key):
        """Evaluate a caller-prepared scenario with this runner's collection contract."""
        return collector(
            state, nnx.state(self.agent.actor), nnx.state(self.agent.critic),
            nnx.state(self.am) if self.am is not None else None,
            key, self.reference_point,
        )

    def _filename(self, stage):
        return {
            "policy": self.training_state_file_name,
            "adaptation": self.adaptation_module_file_name,
            "pretraining": self.pretraining_state_file_name,
        }[stage]

    def begin(self, stage, mode):
        """Fresh refuses overwrites; resume restores optimizer, RNG and progress too."""
        if mode not in ("fresh", "resume"):
            raise ValueError("mode must be fresh or resume")
        if mode == "resume":
            self.restore(stage)
            return self.policy_epoch if stage == "policy" else self.adaptation_epoch
        if checkpoint_exists(self.ckpt_dir, self._filename(stage)):
            raise FileExistsError(
                "Checkpoint exists; choose resume or a new run name/directory"
            )
        if stage == "adaptation":
            self.restore("policy")
        return 0

    def save(self, stage):
        metadata = {
            "stage": stage,
            "teacher_source": self.teacher_source,
            "config_id": self.config_id,
            "resolved_config": self.resolved_config,
            "experiment": self.env.run_name,
            "runner_rng": self._rng,
            "env_rng": self.env._rng,
            "policy_epoch": self.policy_epoch,
            "adaptation_epoch": self.adaptation_epoch,
        }
        if stage == "adaptation":
            metadata["teacher_id"] = model_fingerprint(self.agent.actor)
            save_adaptation_module(
                self.am,
                self.ckpt_dir,
                self._filename(stage),
                optimizer=self.am_optimizer,
                rng_key=self.agent.key,
                metadata=metadata,
            )
        else:
            save_trained_modules(
                self.agent,
                self.ckpt_dir,
                self._filename(stage),
                metadata=metadata,
            )

    def restore(self, stage):
        if stage == "adaptation":
            self.restore("policy")
        payload = load_trained_modules(self.ckpt_dir, self._filename(stage))
        metadata = payload["metadata"]
        if metadata["config_id"] != self.config_id:
            # Choosing to load pretraining is not a change to the trained model.
            saved_config = metadata["resolved_config"]
            pretraining_config = {
                **saved_config,
                "rl": {
                    **saved_config["rl"],
                    "use_pretrained": self.training_cfg.use_pretrained,
                },
            }
            if stage != "pretraining" or pretraining_config != self.resolved_config:
                raise ValueError("Checkpoint configuration differs from this run")
        if stage == "adaptation":
            if metadata["teacher_id"] != model_fingerprint(self.agent.actor):
                raise ValueError(
                    "Adaptation checkpoint belongs to a different teacher policy"
                )
            update_module_from_checkpoint_state(self.am, payload["am_model"])
            _restore_optimizer(self.am_optimizer, payload["am_optimizer"])
            self.agent.key = jax.device_put(payload["am_rng"], self.agent.key.sharding)
        else:
            restore_trained_modules(self.agent, self.ckpt_dir, self._filename(stage))
        self.teacher_source = metadata.get("teacher_source")
        self._rng, self.env._rng = metadata["runner_rng"], metadata["env_rng"]
        self.policy_epoch, self.adaptation_epoch = int(metadata["policy_epoch"]), int(
            metadata["adaptation_epoch"]
        )

    def learn(self, *, mode="fresh"):
        return learn_runner(self, mode=mode)

    def initialize_from_teacher(self, checkpoint):
        return initialize_from_teacher(self, checkpoint)

    def train_adaptation_module_on_policy(self, *, mode="fresh"):
        return train_adaptation_module_on_policy_runner(self, mode=mode)

    def evaluate(self, phase=2, *, allow_privileged_context=False, **kwargs):
        return evaluate_runner(
            self,
            phase,
            allow_privileged_context=allow_privileged_context,
            **kwargs,
        )

    def pretrain(self, strategy="supervised_learning", *, mode="fresh"):
        if strategy != "supervised_learning":
            raise ValueError("Only supervised_learning pretraining is supported")
        return pretrain(self, mode=mode)
