"""Cross-backend checks for paired, randomized demonstration evaluation."""
from pathlib import Path
from types import SimpleNamespace

from flax import nnx
import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from experiments.demonstration_use_cases.common import load_config
from experiments.demonstration_use_cases.mjx_evaluation import prepare_batch, make_rollout
from experiments.demonstration_use_cases.runtime import NativeEnvironment
from experiments.demonstration_use_cases.task import sample_trial, features


class FeedbackActor(nnx.Module):
    def deterministic_action(self, obs):
        return jnp.clip(0.025 + 0.01 * obs, 0., 0.1)


def test_mjx_matches_native_randomized_closed_loop(tmp_path):
    config = load_config('exp3_rl_robustness', mode='smoke')
    common = config['common']
    env = NativeEnvironment(config, 'astrobee', SimpleNamespace(path=tmp_path))
    samples = [sample_trial(common, 20000 + i,
               dict(mass=[1.25, 1.5], inertia=[1.25, 1.5], thrust=[0.5, 0.75], force=0.05, torque=0.005), env.model.nu) for i in range(2)]
    actor = FeedbackActor()
    model, axes, data = prepare_batch(env, samples)
    steps, decimation = 512, 5
    rollout = make_rollout(actor, axes, common['reference'], env.model.actuator_ctrlrange,
                           steps, decimation)
    poses, velocities, commands = jax.device_get(rollout(
        model, data, jnp.asarray(np.stack([s['thrust_scale'] for s in samples]))))
    for lane, sample in enumerate(samples):
        env.reset(sample)
        env.data.xfrc_applied[env.chaser] = sample['wrench']
        for step in range(steps):
            command = np.asarray(actor.deterministic_action(jnp.asarray(
                features(env.get_obs(), np.asarray(common['reference']))[None])))[0]
            env.data.ctrl[:] = command * sample['thrust_scale']
            for _ in range(decimation):
                mujoco.mj_step(env.model, env.data)
            np.testing.assert_allclose(commands[step, lane], command, atol=2e-5)
            # Float32 MJX accumulates roundoff over 2,560 physics steps against
            # float64 native MuJoCo; allow 0.5 mm, far below the 0.25 m task radius.
            np.testing.assert_allclose(poses[step, lane], env.data.qpos, atol=5e-4)
            np.testing.assert_allclose(velocities[step, lane], env.data.qvel, atol=2e-5)


def test_batched_metrics_and_recordings_match_native(tmp_path):
    import json
    from copy import deepcopy
    from experiments.demonstration_use_cases.common import run_directory
    from experiments.demonstration_use_cases.runtime import evaluate

    config = load_config('exp3_rl_robustness', mode='smoke')
    config['common']['evaluation_distributions'] = {
        'nominal': {'initial': 'fixed'},
        'combined': config['common']['evaluation_distributions']['combined'],
    }
    # Exercise early success and recording truncation in addition to dynamics.
    config['common']['env']['environment'].update(terminal_radius=100., terminal_max_att_error=4.)
    job = dict(controller='ppo', regime='nominal', seed=0, spacecraft='astrobee')
    outputs = []
    for backend in ['mujoco_native', 'mjx']:
        cfg = deepcopy(config)
        cfg['protocol']['evaluation_backend'] = backend
        with run_directory(tmp_path / backend, cfg, job) as run:
            evaluate(cfg, job, run, actor=FeedbackActor())
            outputs.append(run.path)
    rows = [[json.loads(line) for line in (p / 'metrics.jsonl').read_text().splitlines()]
            for p in outputs]
    assert len(rows[0]) == len(rows[1]) == 4
    for native, batched in zip(*rows, strict=True):
        for field in ['success', 'episode_length', 'termination_reason', 'evaluation_seed', 'condition']:
            assert native[field] == batched[field]
        assert batched['episode_length'] == 5
        for field in ['final_position_error', 'final_attitude_error', 'control_effort']:
            np.testing.assert_allclose(native[field], batched[field], atol=2e-5)
    for recording in outputs[0].glob('recordings/*/*.npz'):
        other = outputs[1] / recording.relative_to(outputs[0])
        with np.load(recording) as a, np.load(other) as b:
            np.testing.assert_allclose(a['qpos'], b['qpos'], atol=2e-5)
            np.testing.assert_allclose(a['time'], b['time'], atol=1e-7)
            assert len(b['time']) == 6
