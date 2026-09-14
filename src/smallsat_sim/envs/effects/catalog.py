"""Built-in actuator fault definitions and validation."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable
import math

import numpy as np

from .custom import CustomEffectState, apply_custom_effect
from .actuator_kernels import (
    PerturbationStatus, stuck_off_apply_from_state, stuck_on_apply_from_state, gp_apply_from_state,
)


@dataclass(frozen=True)
class FaultSpec:
    effect: str
    actuator: str | None = None
    start_time: float = 0.0
    env_ids: tuple[int, ...] | None = None
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True)
class BuiltinFaultDefinition:
    status: PerturbationStatus
    adapter: str
    kernel: Callable
    accepts_bounds: bool = False

    @property
    def name(self):
        return self.status.name.lower()

    @property
    def status_id(self):
        return self.status.value


# Order preserves mixture RNG assignment and checkpoint tuple order.
BUILTIN_FAULTS = (
    BuiltinFaultDefinition(PerturbationStatus.STUCK_OFF, "StuckOffThrusters", stuck_off_apply_from_state),
    BuiltinFaultDefinition(PerturbationStatus.STUCK_ON, "StuckOnThrusters", stuck_on_apply_from_state),
    BuiltinFaultDefinition(PerturbationStatus.FAULTY_VALVE, "FaultyValve", gp_apply_from_state, True),
    BuiltinFaultDefinition(PerturbationStatus.SATURATED_THRUST, "SaturatedThrust", gp_apply_from_state, True),
    BuiltinFaultDefinition(PerturbationStatus.THRUST_INSTABILITY, "ThrustInstability", gp_apply_from_state, True),
)
FAULT_NAMES = tuple(fault.name for fault in BUILTIN_FAULTS)
FAULTS_BY_NAME = {fault.name: fault for fault in BUILTIN_FAULTS}
FAULTS_BY_STATUS = {fault.status_id: fault for fault in BUILTIN_FAULTS}


_GP_PROFILES = {
    "SATURATED_THRUST": (0.1, 0.01, 1e-4),
    "FAULTY_VALVE": (0.3, 0.4, 5e-4),
    "THRUST_INSTABILITY": (0.15, 0.5, 5e-4),
}


def validate_fault(spec, vehicle, num_envs):
    """Validate scheduling inputs and resolve a physical actuator name."""
    if spec.effect not in FAULT_NAMES:
        raise ValueError(f"Unknown fault {spec.effect!r}; choose from {FAULT_NAMES}")
    if not math.isfinite(spec.start_time) or spec.start_time < 0:
        raise ValueError("Fault start_time must be finite and nonnegative")
    ids = range(num_envs) if spec.env_ids is None else spec.env_ids
    if any(not isinstance(i, (int, np.integer)) or i < 0 or i >= num_envs for i in ids):
        raise ValueError("Fault env_ids must be valid environment indices")
    if len(set(ids)) != len(ids):
        raise ValueError("Fault env_ids must not contain duplicates")
    if (spec.minimum is None) != (spec.maximum is None):
        raise ValueError("Specify both minimum and maximum, or neither")
    if spec.minimum is not None:
        if not FAULTS_BY_NAME[spec.effect].accepts_bounds:
            raise ValueError("minimum/maximum apply only to GP fault modes")
        if not all(map(math.isfinite, (spec.minimum, spec.maximum))) or spec.minimum > spec.maximum:
            raise ValueError("Fault bounds must be finite with minimum <= maximum")
    if spec.actuator is None:
        return None
    names = [actuator.name for actuator in vehicle.actuators]
    if spec.actuator not in names:
        raise ValueError(f"Unknown actuator {spec.actuator!r}; available names: {names}")
    return names.index(spec.actuator)


def configured_faults(values, vehicle, num_envs):
    """Validate the complete schedule before allocating or changing fault state."""
    specs = tuple(value if isinstance(value, FaultSpec) else FaultSpec(**value) for value in values)
    occupied = set()
    for spec in specs:
        index = validate_fault(spec, vehicle, num_envs)
        if index is not None:
            ids = range(num_envs) if spec.env_ids is None else spec.env_ids
            targets = {(env_id, index) for env_id in ids}
            if occupied & targets:
                raise ValueError("An actuator can have only one scheduled fault per environment")
            occupied.update(targets)
    return specs


def fault_weights(distribution):
    """Validate named relative weights and order them for the built-in sampler."""
    if not isinstance(distribution, Mapping):
        raise ValueError(
            "failure_distribution must map effect names to weights, e.g. "
            "{'stuck_off': 1.0}; positional lists are no longer supported"
        )
    unknown = set(distribution) - set(FAULT_NAMES)
    if unknown:
        raise ValueError(f"Unknown fault names {sorted(unknown)}; choose from {FAULT_NAMES}")
    try:
        weights = np.asarray([distribution.get(name, 0.0) for name in FAULT_NAMES], dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Fault weights must be finite nonnegative numbers") from error
    if (weights.shape != (len(FAULT_NAMES),) or not np.all(np.isfinite(weights))
            or np.any(weights < 0) or not np.isfinite(weights.sum()) or weights.sum() <= 0):
        raise ValueError("Fault weights must be finite, nonnegative and have a positive sum")
    return weights


def gp_failure_modes(status_type, *, with_jitter=False):
    """Fresh kernel settings keyed by the backend's existing status enum."""
    return {
        status_type[name]: {
            "kernel": "Matern", "lengthscale": lengthscale, "outputscale": outputscale,
            **({"jitter": jitter} if with_jitter else {}),
        }
        for name, (lengthscale, outputscale, jitter) in _GP_PROFILES.items()
    }


def apply_actuator_effect(state, control, time):
    if state is None:
        return control, state
    if isinstance(state, CustomEffectState):
        return apply_custom_effect(state, control, time)
    definition = FAULTS_BY_STATUS.get(state.failure_value)
    if definition is not None:
        return definition.kernel(state, control, time)
    # Older snapshots without a recognized mode used mask-based stuck faults.
    control, state = stuck_off_apply_from_state(state, control, time)
    return stuck_on_apply_from_state(state, control, time)


def apply_actuator_effects(states, control, time):
    updated = []
    for state in states:
        control, state = apply_actuator_effect(state, control, time)
        updated.append(state)
    return control, tuple(updated)
