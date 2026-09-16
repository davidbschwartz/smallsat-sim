"""Physical vehicle definitions and validated YAML loading, independent of simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from smallsat_sim.configuration import read_yaml
from smallsat_sim.errors import ValidationError


@dataclass(frozen=True)
class BodySpec:
    name: str
    pos: tuple[float, ...] = (0.0, 0.0, 0.0)
    euler: tuple[float, ...] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class ActuatorSpec:
    name: str
    pos: tuple[float, ...]
    gear: tuple[float, ...]
    site: str | None = None
    forcerange: tuple[float, float] = (0.0, 1.0)
    ctrlrange: tuple[float, float] = (0.0, 0.0)
    forcelimited: str = "false"
    ctrllimited: str = "false"
    size: float = 0.005


@dataclass(frozen=True)
class GeomSpec:
    name: str
    type: str
    pos: tuple[float, ...] = (0.0, 0.0, 0.0)
    euler: tuple[float, ...] | str = "0 0 0"
    mesh: str | None = None
    asset_scale: tuple[float, ...] = (1.0, 1.0, 1.0)
    size: tuple[float, ...] = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class PhysicalPropertiesSpec:
    length: float
    width: float
    height: float
    mass: float
    diag_inertia: tuple[float, ...]
    com_offset: tuple[float, ...]
    density: float = 1000.0


@dataclass(frozen=True)
class AssetSpec:
    kind: str = "generic"
    meshes: tuple[str, ...] = ()
    materials: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VehicleSpec:
    name: str
    bodies: tuple[BodySpec, ...]
    actuators: tuple[ActuatorSpec, ...]
    geoms: tuple[GeomSpec, ...]
    physical: PhysicalPropertiesSpec
    assets: AssetSpec = field(default_factory=AssetSpec)
    default_env: str | None = None
    default_model: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def validate_vehicle(vehicle: VehicleSpec) -> VehicleSpec:
    if not vehicle.name:
        raise ValidationError("VehicleSpec.name must be non-empty.")
    if not vehicle.bodies:
        raise ValidationError(f"Vehicle {vehicle.name} must define at least one body.")
    if not vehicle.actuators:
        raise ValidationError(
            f"Vehicle {vehicle.name} must define at least one actuator."
        )
    if vehicle.physical.mass <= 0:
        raise ValidationError(f"Vehicle {vehicle.name} mass must be positive.")
    for actuator in vehicle.actuators:
        if len(actuator.pos) != 3:
            raise ValidationError(f"Actuator {actuator.name} pos must be length 3.")
        if len(actuator.gear) not in (3, 6):
            raise ValidationError(
                f"Actuator {actuator.name} gear must be length 3 or 6."
            )
        if len(actuator.forcerange) != 2:
            raise ValidationError(
                f"Actuator {actuator.name} forcerange must be length 2."
            )
    return vehicle


def load_vehicle(path) -> VehicleSpec:
    """Read an asset YAML file; no registry or environment construction is required."""
    data = read_yaml(path)
    vehicle = VehicleSpec(
        name=data["name"],
        bodies=tuple(BodySpec(**item) for item in data["bodies"]),
        actuators=tuple(ActuatorSpec(**item) for item in data["actuators"]),
        geoms=tuple(GeomSpec(**item) for item in data.get("geoms", [])),
        physical=PhysicalPropertiesSpec(**data["physical"]),
        assets=AssetSpec(**data.get("assets", {})),
        default_env=data.get("default_env"),
        default_model=data.get("default_model"),
        metadata=data.get("metadata", {}),
    )
    return validate_vehicle(vehicle)
