"""Explicit MJX scenarios collected through the standard RL rollout and task.

Scenario dictionaries carry initial_qpos (7), initial_qvel (6), mass_scale,
inertia_scale, thrust_scale (actuators), wrench (6, world frame), and optional
identity/seed metadata. Sampling belongs to the caller; evaluation adds no RNG.
"""

from copy import copy
from dataclasses import fields, replace
from time import perf_counter

from flax import nnx
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from smallsat_sim.envs.effects.custom import CustomEffectState
from smallsat_sim.envs.vec_env import mjx_backend
from smallsat_sim.envs.vec_env.types import VecEnvState
from .rollout.collector import make_collector
from .runner_metrics import scenario_metrics


class _ScenarioModels:
    """Prepare private model/data batches without changing the training environment."""

    def __init__(self, model):
        if (model.nq, model.nv, model.nbody) != (7, 6, 2):
            raise ValueError("Explicit MJX scenarios require one free body")
        self.model = copy(model)
        self.body = 1
        self.mass = float(model.body_mass[self.body])
        self.inertia = model.body_inertia[self.body].copy()
        self.data = mujoco.MjData(self.model)
        self.template = None

    def _validate(self, samples):
        for sample in samples:
            for key, shape in [
                ("initial_qpos", (7,)),
                ("initial_qvel", (6,)),
                ("thrust_scale", (self.model.nu,)),
                ("wrench", (6,)),
            ]:
                value = np.asarray(sample[key])
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"{key} must be finite with shape {shape}")
            if not np.isclose(np.linalg.norm(sample["initial_qpos"][3:]), 1):
                raise ValueError("Initial quaternion must have unit norm")
            for key in ("mass_scale", "inertia_scale"):
                if not np.isfinite(sample[key]) or sample[key] <= 0:
                    raise ValueError(f"{key} must be finite and positive")
            if np.any(np.asarray(sample["thrust_scale"]) < 0):
                raise ValueError("thrust_scale must be nonnegative")

    def _reset(self, sample):
        mujoco.mj_resetData(self.model, self.data)
        self.model.body_mass[self.body] = self.mass * sample["mass_scale"]
        self.model.body_inertia[self.body] = self.inertia * sample["inertia_scale"]
        mujoco.mj_setConst(self.model, self.data)
        self.data.qpos[:] = sample["initial_qpos"]
        self.data.qvel[:] = sample["initial_qvel"]
        self.data.xfrc_applied[self.body] = sample["wrench"]
        mujoco.mj_forward(self.model, self.data)

    def prepare_batch(self, samples):
        if self.template is None:
            # Capture nominal constants before applying any randomized scenario.
            model = mjx.put_model(self.model)
            baseline = {
                f.name: np.array(getattr(self.model, f.name), copy=True)
                for f in fields(model)
                if isinstance(getattr(model, f.name), jax.Array) and hasattr(self.model, f.name)
            }
            self.template = model, mjx.make_data(self.model), baseline
        model, template, baseline = self.template
        snapshots, cache = [], {}
        for sample in samples:
            key = (float(sample["mass_scale"]), float(sample["inertia_scale"]))
            if key not in cache:
                self._reset(sample)
                cache[key] = {
                    name: np.array(getattr(self.model, name), copy=True)
                    for name, value in baseline.items()
                    if not np.array_equal(getattr(self.model, name), value)
                }
            snapshots.append(cache[key])
        updates, axis_updates = {}, {}
        for name in set().union(*(s.keys() for s in snapshots)):
            values = [s.get(name, baseline[name]) for s in snapshots]
            if all(np.array_equal(values[0], v) for v in values[1:]):
                updates[name] = jnp.asarray(values[0])
            else:
                updates[name] = jnp.asarray(np.stack(values))
                axis_updates[name] = 0
        axes = jax.tree.map(lambda _: None, model).replace(**axis_updates)
        model = model.replace(**updates)

        def initial(position, velocity, wrench):
            return template.replace(
                qpos=position,
                qvel=velocity,
                xfrc_applied=template.xfrc_applied.at[self.body].set(wrench),
            )

        data = jax.vmap(initial)(
            jnp.asarray(np.stack([s["initial_qpos"] for s in samples])),
            jnp.asarray(np.stack([s["initial_qvel"] for s in samples])),
            jnp.asarray(np.stack([s["wrench"] for s in samples])),
        )
        data = jax.vmap(mjx.forward, in_axes=(axes, 0))(model, data)
        return model, tuple(sorted(axis_updates)), data


def _scale_thrust(data, values, time, dt):
    return values * data["scale"], data


def _record_pose(state):
    return state.mjx_batch.qpos, state.mjx_batch.qvel


def evaluate_scenarios(runner, scenarios, *, source, batch_size=None, steps=None):
    """Return (episodes, timings), with exactly one result per supplied scenario.

    Uses MJX even when training uses freeflyer. No training RNG, state, or
    randomization is consumed. Terminal flags come from the configured task.
    """
    samples = list(scenarios)
    batch_size = runner.env.num_envs if batch_size is None else batch_size
    steps = runner.training_cfg.episode_len if steps is None else steps
    for name, value in (("batch_size", batch_size), ("steps", steps)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not samples:
        raise ValueError("At least one evaluation scenario is required")
    prepared_models = _ScenarioModels(runner.env.model)
    prepared_models._validate(samples)
    reference = np.asarray(runner.reference_point)
    if reference.ndim == 2:
        if not np.allclose(reference, reference[0], rtol=0, atol=1e-7):
            raise ValueError(
                "Explicit scenarios require a shared pose reference across training lanes"
            )
        reference = reference[0]
    config = runner.env.build_step_config()
    dt = config.model_dt * config.control_decimation
    capacity = min(batch_size, len(samples))
    results, timings, collectors = [], [], {}
    started = perf_counter()
    for start in range(0, len(samples), capacity):
        real = samples[start : start + capacity]
        batch = real + [real[-1]] * (capacity - len(real))
        before = perf_counter()
        model, batched_fields, data = prepared_models.prepare_batch(batch)
        thrust = CustomEffectState(
            data={"scale": jnp.asarray(np.stack([s["thrust_scale"] for s in batch]))},
            active_mask=jnp.ones(capacity, dtype=bool),
            start_times=jnp.zeros(capacity),
            apply_path=__name__ + ":_scale_thrust",
            apply_fn=_scale_thrust,
            dt=dt,
        )
        state = VecEnvState(
            rng=jax.random.key(0),
            mjx_batch=data,
            terminal_hold_counts=jnp.zeros(capacity, dtype=jnp.int32),
            perturbation_states=(thrust,),
        )
        if batched_fields not in collectors:
            collectors[batched_fields] = make_collector(
                runner.env,
                runner.agent,
                runner.am,
                steps=steps,
                context_source=source,
                stochastic=False,
                single_episode=True,
                backend=mjx_backend,
                record_state_fn=_record_pose,
                step_config=replace(config, num_envs=capacity, batched_model_fields=batched_fields),
            )
        ready = perf_counter()
        rollout = collectors[batched_fields](
            state,
            nnx.state(runner.agent.actor),
            None,
            nnx.state(runner.am) if runner.am is not None else None,
            jax.random.key(0),
            jnp.asarray(reference),
            model,
        )
        # Transfer only episode metrics and recordings, not training diagnostics.
        arrays = jax.device_get((rollout.step_outputs, rollout.actions, rollout.trajectory))
        finished = perf_counter()
        rows = scenario_metrics(
            *arrays, initial_poses=np.stack([s["initial_qpos"] for s in batch]), dt=dt
        )
        results.extend(dict(**row, scenario=sample) for row, sample in zip(rows, real))
        timings.append(
            dict(
                scenarios=len(real),
                lanes=capacity,
                preparation_seconds=ready - before,
                rollout_seconds=finished - ready,
                reduction_seconds=perf_counter() - finished,
            )
        )
    return results, dict(
        backend="mjx",
        device=jax.default_backend(),
        scenarios=len(samples),
        batch_size=capacity,
        batches=timings,
        wall_seconds=perf_counter() - started,
        timing_convention="Rollout includes compilation and synchronized device transfer; excludes artifact writes.",
    )
