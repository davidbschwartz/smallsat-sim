"""Public API validation and registry ergonomics tests."""

import pytest

from smallsat_sim.api import (
    ActuatorSpec,
    BodySpec,
    GeomSpec,
    PhysicalPropertiesSpec,
    RegistryError,
    ValidationError,
    VehicleSpec,
    register_vehicle,
    validate_vehicle,
)


def _vehicle(name: str = "test_vehicle") -> VehicleSpec:
    return VehicleSpec(
        name=name,
        bodies=(BodySpec(name="body0"),),
        actuators=(
            ActuatorSpec(
                name="thruster0",
                pos=(0.0, 0.0, 0.0),
                gear=(1.0, 0.0, 0.0),
            ),
        ),
        geoms=(GeomSpec(name="body", type="box"),),
        physical=PhysicalPropertiesSpec(
            length=1.0,
            width=1.0,
            height=1.0,
            mass=1.0,
            diag_inertia=(1.0, 1.0, 1.0),
            com_offset=(0.0, 0.0, 0.0),
        ),
    )


def test_vehicle_registry_rejects_duplicate_names() -> None:
    vehicle = _vehicle("duplicate_vehicle")
    register_vehicle(vehicle, replace=True)
    with pytest.raises(RegistryError):
        register_vehicle(vehicle)


def test_validate_vehicle_catches_shape_errors() -> None:
    invalid = VehicleSpec(
        name="bad_vehicle",
        bodies=(BodySpec(name="body0"),),
        actuators=(ActuatorSpec(name="bad", pos=(0.0, 0.0), gear=(1.0, 0.0, 0.0)),),
        geoms=(),
        physical=PhysicalPropertiesSpec(
            length=1.0,
            width=1.0,
            height=1.0,
            mass=1.0,
            diag_inertia=(1.0, 1.0, 1.0),
            com_offset=(0.0, 0.0, 0.0),
        ),
    )
    with pytest.raises(ValidationError, match="pos must be length 3"):
        validate_vehicle(invalid)
