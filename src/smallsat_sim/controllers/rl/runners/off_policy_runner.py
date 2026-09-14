"""SAC lifecycle, persistent collection, and explicit full/policy checkpoints."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import time

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import wandb

from ..algorithms.sac import SAC
from ..config import validate_policy_context, validate_sac_config, config_dict
from ..storage.replay_buffer import empty_replay, insert, sample, ReplayState, Transition
from .rollout.off_policy_effects import batched_effect_state
from .rollout.off_policy import initial_carry, make_off_policy_collector
from .runner_metrics import report, rollout_metrics
from .runner_setup import build_checkpoint_file_names, configure_jax_compilation_cache
from .runner_utils import _save, load_trained_modules, checkpoint_exists, _checked_state
from smallsat_sim.envs.effects.scheduling import reset_and_randomize


def _checkpoint_count(value, name):
    array = np.asarray(value)
    if array.shape != () or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f'Checkpoint {name} must be an integer scalar')
    return int(array)


class OffPolicyRunner:
    def __init__(self, env, planner, *, config):
        validate_sac_config(config, env.num_envs)
        validate_policy_context(env.env_cfg, config)
        configure_jax_compilation_cache(jax)
        self.env, self.planner, self.training_cfg = env, planner, deepcopy(config)
        self._rng, key, self.replay_key = jax.random.split(env.next_rng_keys(1)[0], 3)
        self.agent = SAC(env, planner, key, config=config)
        self.am = None
        self.reference_point = planner.get_reference(env.get_obs())
        self.context_scale = jnp.asarray(config.context_scale)[:0]
        self.ckpt_dir = str(config.checkpoint_dir)
        for name, value in build_checkpoint_file_names(env, 'sac').items():
            setattr(self, name, value)
        self.replay = None
        self.carry = None
        self.effect_pool = None
        self.transitions = self.gradient_updates = self.policy_epoch = 0
        self.adaptation_epoch = 0
        self._collectors = {}
        self._collector_modes = {}
        self.evaluation_checkpoint = None
        self.resolved_config = {
            'rl': {**config_dict(env.env_cfg.environment), **config_dict(config)},
            'bodies': config_dict(env.env_cfg.Bodies), 'seed': int(env.env_cfg.sim.seed),
            'model': str(env.env_cfg.model), 'physics_dt': float(env.model.opt.timestep),
            'physical_properties': config_dict(env.model_cfg.physical),
            'thruster_mixer': np.asarray(env._thruster_mixer_T).tolist(),
            'actuator_ranges': np.asarray(self.agent.actor.act_range).tolist(),
            'actuator_low': np.asarray(self.agent.actor.act_low).tolist(),
            'actuator_high': np.asarray(self.agent.actor.act_high).tolist(),
            'reference': np.asarray(self.reference_point).tolist(),
        }
        self.config_id = hashlib.sha256(json.dumps(self.resolved_config, sort_keys=True).encode()).hexdigest()
        if env.use_wandb:
            wandb.init(project='Astrobee-training', name=env.run_name, config=self.resolved_config)

    def _take_keys(self):
        self._rng, key = jax.random.split(self._rng)
        return key

    def _initial_carry(self, randomize):
        reset_and_randomize(self.env, self._take_keys(), enabled=randomize)
        state = (self.env.freeflyer_state_struct()
                 if self.env.env_cfg.environment.rollout_backend == 'freeflyer'
                 else self.env.state_struct)
        state = batched_effect_state(self.env, state)
        return initial_carry(state, self.env.rollout_backend(), self.reference_point,
                             self._take_keys(), self.env.num_envs)

    def collector(self, source='zero', *, stochastic=False, steps=None, mode='evaluation'):
        if source not in ('zero', 'estimated', 'privileged'):
            raise ValueError('Unknown context source')
        steps = steps or self.training_cfg.SAC.collection_steps
        cache_key = (steps, stochastic, mode)
        if cache_key not in self._collectors:
            visual = getattr(self.env, "_rl_visualization", None)
            self._collectors[cache_key] = make_off_policy_collector(
                self.env, self.agent.actor, steps=steps, max_ep_len=self.agent.max_ep_len,
                stochastic=stochastic, visualization=visual)
        collector = self._collectors[cache_key]
        self._collector_modes[collector] = mode
        return collector

    def collect(self, collector, *, randomize):
        # Evaluation is isolated from the persistent training carry.
        visual = self.env.rollout_visualization(self._collector_modes[collector], self.env.build_step_config())
        carry = self._initial_carry(randomize)
        _, result = collector(carry, nnx.state(self.agent.actor), self.reference_point, 0, 0)
        self._finish_visualization(visual, result)
        return result

    @staticmethod
    def _finish_visualization(visual, result):
        if visual is not None:
            jax.block_until_ready(result.actions)
            jax.effects_barrier()
            visual.finish()

    def collect_state(self, collector, state, key):
        state = batched_effect_state(self.env, state)
        carry = initial_carry(state, self.env.rollout_backend(), self.reference_point,
                              key, self.env.num_envs)
        _, result = collector(carry, nnx.state(self.agent.actor), self.reference_point, 0, 0)
        return result

    def _make_effect_pool(self):
        """Prepare a finite seeded pool once; reset sampling stays on the device."""
        if not self.env.train_with_failures:
            return None
        candidates = []
        count = (self.training_cfg.SAC.randomization_pool_size + self.env.num_envs - 1) // self.env.num_envs
        for _ in range(count):
            reset_and_randomize(self.env, self._take_keys(), enabled=True)
            state = batched_effect_state(self.env, self.env.state_struct)
            candidates.append((state.disturbance_states, state.perturbation_states))
        # RNG keys belong to effect adapters, not individual rows. Kernels do not
        # sample from them; preserve a single key while concatenating row data.
        return jax.tree_util.tree_map_with_path(
            lambda path, *x: x[0] if getattr(path[-1], 'name', None) == 'rng'
            else jnp.concatenate(x, axis=0), *candidates)

    def learn(self, *, mode='fresh'):
        hp = self.training_cfg.SAC
        if mode not in ('fresh', 'resume'):
            raise ValueError('mode must be fresh or resume')
        if mode == 'resume':
            self.restore('policy')
        else:
            if checkpoint_exists(self.ckpt_dir, self.training_state_file_name):
                raise FileExistsError('Checkpoint exists; choose resume or a new run name/directory')
            self.replay = empty_replay(hp.replay_capacity, self.env.obs_dim, self.env.act_dim)
            self.carry = self._initial_carry(self.env.train_with_failures)
            self.effect_pool = self._make_effect_pool()
            self.transitions = self.gradient_updates = self.policy_epoch = 0
        collector = self.collector(stochastic=True, mode='training')
        chunk = hp.collection_steps * self.env.num_envs
        while self.transitions < hp.total_transitions:
            start = time.perf_counter()
            visual = self.env.rollout_visualization("training", self.env.build_step_config())
            self.carry, result = collector(self.carry, nnx.state(self.agent.actor),
                                          self.reference_point, self.transitions, hp.random_steps,
                                          self.effect_pool)
            self._finish_visualization(visual, result)
            flat = jax.tree.map(lambda x: x.reshape((-1, *x.shape[2:])), result.transitions)
            self.replay = insert(self.replay, flat)
            self.transitions += chunk
            self.policy_epoch += 1
            metrics = rollout_metrics(result)
            if self.transitions >= hp.learning_starts:
                updates = []
                for _ in range(hp.updates_per_collection):
                    self.replay_key, key = jax.random.split(self.replay_key)
                    updates.append(self.agent.update(sample(self.replay, key, hp.batch_size)))
                    self.gradient_updates += 1
                metrics.update(jax.tree.map(lambda *x: jnp.mean(jnp.stack(x)), *updates))
            metrics.update(environment_transitions=self.transitions, gradient_updates=self.gradient_updates,
                           replay_size=self.replay.size,
                           updates_per_transition=hp.updates_per_collection / chunk)
            jax.block_until_ready(metrics)
            metrics['epoch_seconds'] = time.perf_counter() - start
            report(self, 'policy_training', self.policy_epoch, metrics)
            if self.policy_epoch % self.training_cfg.training_checkpoint_interval == 0:
                self.save('policy')
        self.save('policy')

    @property
    def policy_file_name(self):
        name = self.training_state_file_name
        return name if name.endswith('_actor') else name + '_actor'

    def save(self, stage='policy'):
        if stage != 'policy' or self.carry is None or self.replay is None:
            raise ValueError('SAC saves initialized policy training only')
        metadata = dict(stage='policy', algorithm='sac', config_id=self.config_id,
                        resolved_config=self.resolved_config, experiment=self.env.run_name,
                        policy_epoch=self.policy_epoch, adaptation_epoch=0)
        actor = {'actor_model': nnx.state(self.agent.actor), 'metadata': metadata}
        # Store dynamic environment/carry leaves against a runtime template; this
        # covers both physics backends and custom effect PyTrees without new codecs.
        payload = dict(actor, sac_version=1, models=nnx.state(self.agent.objects()),
                       carry_leaves=jax.tree.leaves(self.carry),
                       effect_pool_leaves=jax.tree.leaves(self.effect_pool),
                       carry_paths=[jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_flatten_with_path(self.carry)[0]],
                       replay=self.replay._asdict(), runner_rng=self._rng,
                       env_rng=self.env._rng, replay_rng=self.replay_key, agent_rng=self.agent.key,
                       transitions=self.transitions, gradient_updates=self.gradient_updates)
        payload['replay'] = dict(data=self.replay.data._asdict(), cursor=self.replay.cursor, size=self.replay.size)
        _save(self.ckpt_dir, self.training_state_file_name, payload)
        _save(self.ckpt_dir, self.policy_file_name, actor)

    def restore(self, stage='policy'):
        if stage != 'policy':
            raise ValueError('SAC does not support adaptation or pretraining')
        payload = load_trained_modules(self.ckpt_dir, self.training_state_file_name)
        if payload.get('sac_version') != 1 or payload['metadata']['config_id'] != self.config_id:
            raise ValueError('Checkpoint configuration differs from this SAC run')
        states = _checked_state(nnx.state(self.agent.objects()), payload['models'])
        state = (self.env.freeflyer_state_struct()
                 if self.env.env_cfg.environment.rollout_backend == 'freeflyer'
                 else self.env.state_struct)
        state = batched_effect_state(self.env, state)
        template = initial_carry(state, self.env.rollout_backend(), self.reference_point,
                                 self._rng, self.env.num_envs)
        paths = [jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_flatten_with_path(template)[0]]
        if paths != payload['carry_paths']:
            raise ValueError('Checkpoint collector structure differs')
        leaves, tree = jax.tree.flatten(template)
        saved = payload['carry_leaves']
        if len(leaves) != len(saved) or any(np.shape(a) != np.shape(b) or a.dtype != b.dtype
                                          for a, b in zip(leaves, saved)):
            raise ValueError('Checkpoint collector state differs')
        carry = jax.tree.unflatten(tree, saved)
        data = Transition(**payload['replay']['data'])
        expected = jax.eval_shape(lambda: empty_replay(
            self.training_cfg.SAC.replay_capacity, self.env.obs_dim, self.env.act_dim))
        if any(a.shape != b.shape or a.dtype != b.dtype for a, b in zip(expected.data, data)):
            raise ValueError('Checkpoint replay dimensions differ')
        pool = None
        if self.env.train_with_failures:
            effects = (template.state.disturbance_states, template.state.perturbation_states)
            paths_and_leaves, effect_tree = jax.tree_util.tree_flatten_with_path(effects)
            pool_leaves = payload['effect_pool_leaves']
            count = ((self.training_cfg.SAC.randomization_pool_size + self.env.num_envs - 1)
                     // self.env.num_envs) * self.env.num_envs
            if len(pool_leaves) != len(paths_and_leaves):
                raise ValueError('Checkpoint effect pool structure differs')
            for (path, leaf), saved_leaf in zip(paths_and_leaves, pool_leaves):
                shape = leaf.shape if getattr(path[-1], 'name', None) == 'rng' else (count, *leaf.shape[1:])
                if saved_leaf.shape != shape or saved_leaf.dtype != leaf.dtype:
                    raise ValueError('Checkpoint effect pool dimensions differ')
            pool = jax.tree.unflatten(effect_tree, pool_leaves)
        capacity = self.training_cfg.SAC.replay_capacity
        cursor = _checkpoint_count(payload['replay']['cursor'], 'replay cursor')
        size = _checkpoint_count(payload['replay']['size'], 'replay size')
        if not 0 <= cursor < capacity or not 0 < size <= capacity:
            raise ValueError('Checkpoint replay cursor or size is invalid')
        transitions = _checkpoint_count(payload['transitions'], 'transitions')
        updates = _checkpoint_count(payload['gradient_updates'], 'gradient updates')
        epoch = _checkpoint_count(payload['metadata']['policy_epoch'], 'policy epoch')
        if min(transitions, updates, epoch) < 0:
            raise ValueError('Checkpoint progress must be nonnegative')
        for name, key in (('runner_rng', self._rng), ('env_rng', self.env._rng),
                          ('replay_rng', self.replay_key), ('agent_rng', self.agent.key)):
            restored = payload[name]
            if restored.shape != key.shape or restored.dtype != key.dtype:
                raise ValueError(f'Checkpoint RNG layout differs at {name}')
        nnx.update(self.agent.objects(), states)
        self.carry = carry
        self.effect_pool = pool
        self.replay = ReplayState(data, jnp.array(cursor, jnp.int32), jnp.array(size, jnp.int32))
        self._rng, self.env._rng = payload['runner_rng'], payload['env_rng']
        self.replay_key, self.agent.key = payload['replay_rng'], payload['agent_rng']
        self.transitions, self.gradient_updates = transitions, updates
        self.policy_epoch = epoch

    def evaluate(self, phase=2, *, randomize=None, **kwargs):
        if phase not in (1, 2):
            raise ValueError('Phase must be 1 or 2')
        self.restore_for_evaluation('zero')
        collector = self.collector(steps=self.training_cfg.episode_len)
        return [report(self, 'evaluation_zero', i + 1, rollout_metrics(self.collect(
            collector, randomize=self.env.train_with_failures if randomize is None else randomize)))
            for i in range(self.training_cfg.n_evals)]

    def restore_for_evaluation(self, source='zero'):
        if self.evaluation_checkpoint is not None:
            path = Path(self.evaluation_checkpoint)
        else:
            filename = self.policy_file_name
            if not checkpoint_exists(self.ckpt_dir, filename):
                filename = self.training_state_file_name
            path = Path(self.ckpt_dir) / filename
        payload = load_trained_modules(path.parent, path.name)
        if payload['metadata']['config_id'] != self.config_id:
            raise ValueError('Checkpoint configuration differs from this SAC run')
        nnx.update(self.agent.actor, _checked_state(nnx.state(self.agent.actor), payload['actor_model']))

    def pretrain(self, *args, **kwargs):
        raise ValueError('SAC pretraining is not supported')

    def train_adaptation_module_on_policy(self, *args, **kwargs):
        raise ValueError('SAC adaptation is not supported')

    def initialize_from_teacher(self, *args, **kwargs):
        raise ValueError('SAC teacher initialization is not supported')
