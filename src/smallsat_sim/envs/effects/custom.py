"""Import, validate and execute user-defined environment effects."""

from dataclasses import dataclass, replace
from importlib import import_module
import math
from typing import Any, Callable, Protocol, cast

import jax
import jax.numpy as jnp


EffectData = dict[str, Any]


@dataclass(frozen=True)
class EffectSampleContext:
    """Host-side physical information available when sampling a fresh realization."""
    vehicle: object
    num_envs: int
    dt: float


class SampleEffect(Protocol):
    def __call__(
        self, key: jax.Array, context: EffectSampleContext, **params: Any
    ) -> EffectData:
        """Sample on the host; every array leaf must have context.num_envs rows."""
        ...


class ApplyEffect(Protocol):
    def __call__(
        self, data: EffectData, values: jax.Array, time: jax.Array | float, dt: float
    ) -> tuple[jax.Array, EffectData]:
        """Pure JAX operation preserving value and state shapes and dtypes."""
        ...


@dataclass(frozen=True)
class CustomEffectState:
    """Dynamic sampled arrays plus static implementation identity for JAX/replay."""
    data: EffectData
    active_mask: jax.Array
    start_times: jax.Array
    apply_path: str
    dt: float
    # The callable is static PyTree metadata: imports never execute inside a scan.
    apply_fn: ApplyEffect
    version: str = "1"

    def replace(self, **kwargs):
        return replace(self, **kwargs)


jax.tree_util.register_dataclass(
    CustomEffectState,
    data_fields=["data", "active_mask", "start_times"],
    meta_fields=["apply_path", "dt", "apply_fn", "version"],
)


class CustomEffect:
    """Imperative facade for the same state and function used by compiled stepping."""
    def __init__(self, spec: dict, context: EffectSampleContext, key: jax.Array,
                 *, disturbance: bool = False) -> None:
        sample_key, activation_key = jax.random.split(key)
        sample = cast(SampleEffect, import_function(spec["sample"]))
        data = _sample_batched_data(sample, sample_key, context, spec.get("params", {}))
        self.state = CustomEffectState(
            data=data,
            active_mask=jax.random.bernoulli(
                activation_key, float(spec.get("probability", 1.0)), (context.num_envs,)
            ),
            start_times=jnp.full((context.num_envs,), float(spec.get("start_time", 0.0))),
            apply_path=spec["apply"],
            dt=context.dt,
            apply_fn=cast(ApplyEffect, import_function(spec["apply"])),
            version=str(spec.get("version", "1")),
        )
        self.disturbance = disturbance
        self.num_envs = context.num_envs
        if disturbance:
            self.value_shape = (context.num_envs, 6)
        else:
            self.value_shape = (context.num_envs, len(context.vehicle.actuators))
        _validate_apply_contract(self.state, self.value_shape)

    def restore_state(self, state):
        self.state = state

    def apply_actuator(self, values, time=0.0):
        output, self.state = apply_custom_effect(self.state, values, time)
        return output

    def apply_wrench(self, time=0.0):
        return self.apply_actuator(jnp.zeros(self.value_shape), time)

    def apply(self, values=None, timestamp=0.0):
        if self.disturbance:
            return self.apply_wrench(timestamp if values is None else values)
        return self.apply_actuator(values, timestamp)


def append_custom_effects(env, *, disturbance):
    setting = "custom_disturbances" if disturbance else "custom_faults"
    specs = getattr(env.env_cfg.environment, setting, ())
    if not specs:
        return
    if disturbance:
        effects = env.disturbances.disturbances
    else:
        effects = env.perturbations.perturbations
    context = EffectSampleContext(
        vehicle=env.model_cfg,
        num_envs=env.num_envs,
        dt=float(env.model.opt.timestep) * env.env_cfg.environment.control_decimation,
    )
    for spec in specs:
        effect = CustomEffect(
            spec, context, env.next_rng_keys(1)[0], disturbance=disturbance,
        )
        effects.append(effect)


def apply_custom_effect(
    state: CustomEffectState, values: jax.Array, time: jax.Array | float
) -> tuple[jax.Array, CustomEffectState]:
    """Apply active rows and advance their state; inactive rows retain both inputs."""
    proposed_values, proposed_data = state.apply_fn(state.data, values, time, state.dt)
    active_rows = state.active_mask & (time >= state.start_times)

    def update_active_rows(previous, proposed):
        broadcast_shape = (active_rows.shape[0],) + (1,) * (previous.ndim - 1)
        return jnp.where(active_rows.reshape(broadcast_shape), proposed, previous)

    updated_values = update_active_rows(values, proposed_values)
    updated_data = jax.tree.map(update_active_rows, state.data, proposed_data)
    return updated_values, state.replace(data=updated_data)


def validate_custom_effects(specs) -> None:
    """Validate configuration and resolve callables before allocating a simulator."""
    for spec in specs:
        unknown = set(spec) - {"sample", "apply", "params", "probability", "start_time", "version"}
        if unknown:
            raise ValueError(f"Unknown custom effect settings: {sorted(unknown)}")
        for field in ("sample", "apply"):
            import_function(spec[field])
        if not 0 <= spec.get("probability", 1.) <= 1:
            raise ValueError("Custom effect probability must be in [0, 1]")
        start = spec.get("start_time", 0.)
        if not math.isfinite(start) or start < 0:
            raise ValueError("Custom effect start_time must be finite and nonnegative")


def validate_custom_replay(expected_states, restored_states):
    """Reject changed custom implementations or layouts before adopting a snapshot."""
    for expected, restored in zip(expected_states, restored_states, strict=True):
        expected_custom = isinstance(expected, CustomEffectState)
        restored_custom = isinstance(restored, CustomEffectState)
        if not expected_custom and not restored_custom:
            continue
        if (expected_custom != restored_custom
                or _implementation_identity(expected) != _implementation_identity(restored)):
            raise ValueError(
                "Custom effect replay requires the same apply function, version and timestep"
            )
        if _state_array_layout(expected) != _state_array_layout(restored):
            raise ValueError(
                "Custom effect replay state layout does not match the configured effect"
            )


def import_function(path: str) -> Callable:
    module, separator, name = path.partition(":")
    if not separator or not module or not name:
        raise ValueError("Effect functions must use 'importable.module:function' paths")
    function = getattr(import_module(module), name)
    if not callable(function):
        raise ValueError(f"Effect function {path!r} is not callable")
    return function


def _sample_batched_data(sample: SampleEffect, key: jax.Array,
                         context: EffectSampleContext, params: dict) -> EffectData:
    data = sample(key, context, **params)
    if not isinstance(data, dict):
        raise ValueError("Effect sample must return a dictionary of batched arrays")
    data = jax.tree.map(jnp.asarray, data)
    for leaf in jax.tree.leaves(data):
        if leaf.ndim == 0 or leaf.shape[0] != context.num_envs:
            raise ValueError(
                "Each effect state array must have num_envs as its first dimension"
            )
    return data


def _validate_apply_contract(state: CustomEffectState, value_shape: tuple[int, int]) -> None:
    """Trace once at construction; reject incompatible effects before a rollout."""
    values = jax.ShapeDtypeStruct(value_shape, jnp.float32)
    time = jax.ShapeDtypeStruct((value_shape[0],), jnp.float32)
    output, updated = jax.eval_shape(state.apply_fn, state.data, values, time, state.dt)
    if output.shape != values.shape or output.dtype != values.dtype:
        raise ValueError("Effect apply must preserve input shape and dtype")
    expected_layout = _array_layout(state.data)
    updated_layout = _array_layout(updated)
    if expected_layout != updated_layout:
        raise ValueError("Effect apply must preserve state structure, shapes and dtypes")


def _state_array_layout(state):
    """Compare shapes and dtypes without comparing sampled array values."""
    return _array_layout((state.data, state.active_mask, state.start_times))


def _implementation_identity(state):
    return state.apply_path, state.version, state.dt


def _array_layout(arrays):
    return jax.tree.map(lambda array: (array.shape, array.dtype), arrays)


def __getattr__(name):
    # Keep old checkpoint imports lazy.
    if name in ('custom_state_to_serializable', 'custom_state_from_serializable'):
        from . import serialization
        return getattr(serialization, name)
    raise AttributeError(name)
