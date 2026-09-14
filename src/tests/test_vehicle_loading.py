"""Physical loading stays independent; existing public imports remain compatible."""

import pickle
import subprocess
import sys

import pytest

from smallsat_sim.model import vehicle
from smallsat_sim.errors import ValidationError


def test_model_loading_does_not_initialize_behavior_or_simulation():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from smallsat_sim.model.vehicle import load_vehicle

assert load_vehicle('vehicles/astrobee.yaml').physical.mass > 0
unexpected = [name for name in sys.modules if name.startswith((
    'smallsat_sim.api', 'smallsat_sim.envs', 'smallsat_sim.controllers',
))]
assert not unexpected, unexpected
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "name",
    ["BodySpec", "ActuatorSpec", "GeomSpec", "PhysicalPropertiesSpec", "AssetSpec", "VehicleSpec"],
)
def test_public_types_are_the_canonical_model_types(name):
    from smallsat_sim import api

    canonical = getattr(vehicle, name)
    assert getattr(api, name) is canonical


def test_existing_loader_and_validation_imports_are_canonical():
    from smallsat_sim import api
    from smallsat_sim.api import validation, vehicles

    assert api.load_vehicle is vehicles.load_vehicle is vehicle.load_vehicle
    assert api.validate_vehicle is validation.validate_vehicle is vehicle.validate_vehicle
    assert api.ValidationError is validation.ValidationError is ValidationError
    assert pickle.loads(b"csmallsat_sim.api.validation\nValidationError\n.") is ValidationError


@pytest.mark.parametrize(
    "name",
    ["astrobee", "cubesat", "sprint"],
)
def test_packaged_vehicles_load_independent_physical_definitions(name):
    definition = vehicle.load_vehicle(f"vehicles/{name}.yaml")
    second = vehicle.load_vehicle(f"vehicles/{name}.yaml")
    assert definition == second
    assert definition is not second
    assert definition.physical is not second.physical
    assert definition.actuators[0] is not second.actuators[0]


def test_environment_keeps_supplied_vehicle_and_owns_episode_bodies(monkeypatch):
    from dataclasses import replace
    from smallsat_sim.envs import base_env

    class ConfigLoader:
        _load_cfg = base_env.BaseEnv._load_cfg

    definition = replace(
        vehicle.load_vehicle("vehicles/astrobee.yaml"),
        bodies=(vehicle.BodySpec(name="asset_body", pos=(100.0, 200.0, 300.0)),),
    )

    def unexpected_load(*args):
        raise AssertionError("A supplied physical definition must not load a default asset")

    monkeypatch.setattr(base_env, "load_vehicle", unexpected_load)
    env_cfg, physical = ConfigLoader()._load_cfg("astrobee", vehicle=definition)
    assert physical is definition
    assert env_cfg.Bodies.bodies_list[0].name == "body0"
    assert tuple(env_cfg.Bodies.bodies_list[0].pos) != definition.bodies[0].pos


def test_environment_loads_default_asset_once(monkeypatch):
    from smallsat_sim.envs import base_env

    class ConfigLoader:
        _load_cfg = base_env.BaseEnv._load_cfg

    paths = []

    def load(path):
        paths.append(path)
        return vehicle.load_vehicle(path)

    monkeypatch.setattr(base_env, "load_vehicle", load)
    _, physical = ConfigLoader()._load_cfg("astrobee")
    assert paths == ["vehicles/astrobee.yaml"]
    assert isinstance(physical, vehicle.VehicleSpec)


def test_xml_uses_episode_bodies_and_actuator_name_when_site_is_omitted():
    from dataclasses import replace
    from types import SimpleNamespace
    from smallsat_sim.envs.config import resolve_cubesat_config as resolve_config
    from smallsat_sim.model.mujoco_xml import build_mujoco_xml

    definition = vehicle.load_vehicle("vehicles/cubesat.yaml")
    definition = replace(
        definition,
        bodies=(vehicle.BodySpec(name="asset_body"),),
        actuators=(replace(definition.actuators[0], name="test_thruster", site=None),),
    )
    env_cfg = resolve_config()
    env_cfg.Bodies = SimpleNamespace(
        bodies_list=[SimpleNamespace(name="episode_body", pos=[1, 2, 3], euler=[0, 0, 0])],
        num_bodies=1,
    )
    xml = build_mujoco_xml(env_cfg, definition)
    assert "episode_body_test_thruster" in xml
    assert "asset_body" not in xml
    assert "episode_body_None" not in xml
