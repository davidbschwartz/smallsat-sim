"""Batched MJX evaluation with the native evaluator's exact sampled scenarios.

Each lane runs one episode without resets. Host-side metrics truncate at the
first success/failure, retaining the same pose recordings and trial identities.
"""

from time import perf_counter
from dataclasses import fields

import jax
import jax.numpy as jnp
from flax import nnx
from mujoco import mjx
import numpy as np

from smallsat_sim.envs.vec_env.observations import state_features
from smallsat_sim.utils.quaternions_jax import quaternion_to_rotation_matrix

from .task import PoseMetrics, sample_trial


def prepare_batch(env, samples):
    """Compile every randomized native model, sharing unchanged MJX arrays.

    mj_setConst (inside reset) recalculates all mass-dependent constants before
    conversion; scaling only body_mass/body_inertia would miss those constants.
    """
    if not hasattr(env, '_evaluation_mjx_template'):
        model = mjx.put_model(env.model)
        # Keep an immutable host snapshot before reset changes the native model.
        baseline = {f.name: np.array(getattr(env.model, f.name), copy=True)
                    for f in fields(model)
                    if isinstance(getattr(model, f.name), jax.Array)
                    and hasattr(env.model, f.name)}
        env._evaluation_mjx_template = model, mjx.make_data(env.model), baseline
    model, template, baseline = env._evaluation_mjx_template
    snapshots = []
    for sample in samples:
        env.reset(sample)
        snapshots.append({name: np.array(getattr(env.model, name), copy=True)
                          for name, value in baseline.items()
                          if not np.array_equal(getattr(env.model, name), value)})
    updates, axis_updates = {}, {}
    for name in set().union(*(snapshot.keys() for snapshot in snapshots)):
        values = [snapshot.get(name, baseline[name]) for snapshot in snapshots]
        if all(np.array_equal(values[0], value) for value in values[1:]):
            updates[name] = jnp.asarray(values[0])
        else:
            updates[name] = jnp.asarray(np.stack(values))
            axis_updates[name] = 0
    axes = jax.tree.map(lambda _: None, model).replace(**axis_updates)
    model = model.replace(**updates)
    # mjx.step performs forward dynamics itself. Only episode state and the
    # external world wrench differ initially; avoid converting N full models/data.
    def initial(position, velocity, wrench):
        return template.replace(qpos=position, qvel=velocity,
            xfrc_applied=template.xfrc_applied.at[env.chaser].set(wrench))
    data = jax.vmap(initial)(
        jnp.asarray(np.stack([s['initial_qpos'] for s in samples])),
        jnp.asarray(np.stack([s['initial_qvel'] for s in samples])),
        jnp.asarray(np.stack([s['wrench'] for s in samples])))
    return model, axes, data


def make_rollout(actor, model_axes, reference, limits, steps, decimation):
    reference, limits = jnp.asarray(reference), jnp.asarray(limits)
    advance = jax.vmap(mjx.step, in_axes=(model_axes, 0))

    @nnx.jit
    def rollout(policy, model, data, thrust):
        def step(current, _):
            rotation = quaternion_to_rotation_matrix(current.qpos[:, 3:7])
            velocity = jnp.einsum('bji,bj->bi', rotation, current.qvel[:, :3])
            obs = state_features(current.qpos, velocity, current.qvel[:, 3:6], reference)
            command = jnp.clip(policy.deterministic_action(obs), limits[:, 0], limits[:, 1])
            current = current.replace(ctrl=jnp.nan_to_num(command) * thrust)
            following = jax.lax.fori_loop(0, decimation, lambda _, d: advance(model, d), current)
            return following, (following.qpos, following.qvel, command)
        return jax.lax.scan(step, data, None, length=steps)[1]

    return lambda model, data, thrust: rollout(actor, model, data, thrust)


def evaluate_mjx(config, job, run, actor):
    from .runtime import NativeEnvironment

    start = perf_counter()
    common, protocol = config['common'], config['protocol']
    env = NativeEnvironment(config, job['spacecraft'], run)
    reference = np.asarray(common['reference'])
    decimation = common['env']['environment']['control_decimation']
    dt = common['env']['sim']['dt'] * decimation
    run.update(task='pose_regulation', evaluation_num_envs=protocol['trials'],
               evaluation_backend='mjx', evaluation_device=jax.default_backend())
    # Each condition is a fixed-size batch; parameters are runtime inputs.
    rollouts = {}
    for condition, distribution in common['evaluation_distributions'].items():
        batch_start = perf_counter()
        samples = [sample_trial(common, common['evaluation_seed_start'] + trial,
                                distribution, env.model.nu)
                   for trial in range(protocol['trials'])]
        for trial, sample in enumerate(samples):
            run.append(dict(condition=condition, trial=trial, **sample), 'samples.jsonl')
        model, axes, data = prepare_batch(env, samples)
        # The vmap axes differ for nominal versus randomized physical models.
        key = tuple(jax.tree.leaves(axes, is_leaf=lambda x: x is None))
        if key not in rollouts:
            rollouts[key] = make_rollout(actor, axes, reference, env.model.actuator_ctrlrange,
                                        common['episode_steps'], decimation)
        poses, velocities, commands = jax.device_get(rollouts[key](
            model, data, jnp.asarray(np.stack([s['thrust_scale'] for s in samples]))))
        batch_seconds = perf_counter() - batch_start
        for trial, sample in enumerate(samples):
            metrics, reason = PoseMetrics(common), None
            saved = [sample['initial_qpos']]
            for pose, velocity, command in zip(poses[:, trial], velocities[:, trial],
                                                commands[:, trial], strict=True):
                if not np.isfinite(command).all():
                    reason = 'nonfinite_control'
                    break
                obs = np.r_[pose, velocity]
                if not np.isfinite(obs).all():
                    reason = 'nonfinite_state'
                    break
                saved.append(pose)
                if metrics.add(obs, reference, command):
                    break
            path = run.path / 'recordings' / condition / f'{trial:04d}.npz'
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, time=np.arange(len(saved)) * dt, qpos=np.asarray(saved))
            row = metrics.row(reason)
            run.append(dict(**{**job, 'condition': condition}, trial=trial,
                            evaluation_seed=sample['evaluation_seed'],
                            initial_condition_seed=sample['initial_condition_seed'],
                            training_seed=job['seed'], solver_status=0,
                            mean_solve_seconds=None, max_solve_seconds=None,
                            affected_thruster=None, fault_type=condition,
                            evaluation_batch_wall_seconds=batch_seconds,
                            normalized_control_effort=row['control_effort'] /
                            max(float(env.model.actuator_ctrlrange[:, 1].sum()), 1e-12), **row))
        print(f'MJX evaluation {condition}: {len(samples)} episodes in {batch_seconds:.2f}s '
              '(including preparation and compilation)', flush=True)
    run.update(evaluation_wall_seconds=perf_counter() - start)
