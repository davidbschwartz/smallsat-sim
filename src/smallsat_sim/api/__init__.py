"""Asset and reward definitions. See api.experiments for YAML experiment construction."""

from smallsat_sim.envs.effects.custom import EffectSampleContext
from smallsat_sim.api import vec_env_rewards
from smallsat_sim.envs.effects.catalog import FaultSpec, FAULT_NAMES, validate_fault
from smallsat_sim.api.registry import (
    RegistryError,
    describe_vehicle,
    get_reward,
    get_termination,
    register_termination,
    list_terminations,
    get_vehicle,
    list_rewards,
    list_vehicles,
    register_reward,
    register_vehicle,
)
from smallsat_sim.api.rewards import (
    RewardTerm,
    compose_reward,
    RewardCallable,
    RewardContext,
    RewardFunction,
    RewardResult,
)
from smallsat_sim.envs.termination import TerminationContext, TerminationResult
from smallsat_sim.api.runs import RunSpec, build_checkpoint_file_names, build_run_name
from smallsat_sim.model.vehicle import (
    ActuatorSpec,
    AssetSpec,
    BodySpec,
    GeomSpec,
    PhysicalPropertiesSpec,
    VehicleSpec,
)
from smallsat_sim.api.validation import (
    validate_reward,
)
from smallsat_sim.api.vehicles import register_vehicle_file
from smallsat_sim.model.vehicle import load_vehicle, validate_vehicle
from smallsat_sim.errors import ValidationError


def load_builtin_apis() -> None:
    """Built-ins are registered when their modules are imported above."""


__all__ = [
    "EffectSampleContext",
    "FaultSpec", "FAULT_NAMES", "validate_fault",
    "TerminationContext", "TerminationResult",
    "get_termination", "register_termination", "list_terminations",
    "load_vehicle",
    "register_vehicle_file",
    "AssetSpec",
    "ActuatorSpec",
    "BodySpec",
    "GeomSpec",
    "PhysicalPropertiesSpec",
    "RegistryError",
    "RewardTerm",
    "compose_reward",
    "RewardContext",
    "RewardCallable",
    "RewardFunction",
    "RewardResult",
    "RunSpec",
    "ValidationError",
    "VehicleSpec",
    "build_checkpoint_file_names",
    "build_run_name",
    "describe_vehicle",
    "get_reward",
    "get_vehicle",
    "list_rewards",
    "list_vehicles",
    "register_reward",
    "register_vehicle",
    "validate_reward",
    "validate_vehicle",
    "load_builtin_apis",
]
