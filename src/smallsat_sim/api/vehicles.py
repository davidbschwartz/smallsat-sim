"""Optional name registration for YAML-defined physical vehicles."""

from dataclasses import replace as replace_vehicle

from smallsat_sim.api.registry import register_vehicle
from smallsat_sim.model.vehicle import VehicleSpec, load_vehicle


_BUILTIN_ASSETS = {
    "astrobee": "vehicles/astrobee.yaml",
    "astrobee_rl": "vehicles/astrobee.yaml",
    "cubesat": "vehicles/cubesat.yaml",
    "sprint": "vehicles/sprint.yaml",
    "sprint_rl": "vehicles/sprint.yaml",
}


def register_builtin_vehicles(*, replace: bool = True) -> tuple[str, ...]:
    registered = []
    for name, path in _BUILTIN_ASSETS.items():
        vehicle = load_vehicle(path)
        register_vehicle(replace_vehicle(vehicle, name=name), replace=replace)
        registered.append(name)
    return tuple(registered)


def register_vehicle_file(path, *, replace=False) -> VehicleSpec:
    """Make a YAML-defined vehicle available by its name."""
    return register_vehicle(load_vehicle(path), replace=replace)
