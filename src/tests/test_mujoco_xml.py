"""Scene-specific physics stays explicit within the shared XML builder."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from smallsat_sim.configuration import settings
from smallsat_sim.model.mujoco_xml import build_mujoco_xml
from smallsat_sim.model.vehicle import load_vehicle


def scene_config():
    return settings({
        "model": "test_satellite",
        "compiler": {"convexhull": "false"},
        "visual": {"headlight": {"ambient": 0.4, "specular": 0.2, "diffuse": 0.6}},
        "renderer": {"width": 64, "height": 48},
        "sim": {"dt": 0.01},
        "Bodies": {"bodies_list": [
            {"name": "body0", "pos": [1, 2, 3], "euler": [10, 20, 30]},
        ]},
    })


@pytest.mark.parametrize("scene", ["gateway", "training"])
def test_scene_joint_timestep_and_naming_conventions(scene):
    config = scene_config()
    vehicle = load_vehicle("vehicles/astrobee.yaml")
    root = ET.fromstring(build_mujoco_xml(config, vehicle, scene=scene))
    body = root.find("worldbody/body[@name='body0']")
    compiler = root.find("compiler")
    assert "convexhull" not in compiler.attrib
    assert root.find("visual/global").attrib == {"offwidth": "64", "offheight": "48"}
    assert body.find("freejoint") is not None
    assert body.findall("joint") == []
    assert body.get("euler") == "10 20 30"
    if scene == "training":
        assert "eulerseq" not in compiler.attrib
        floor = root.find("worldbody/geom[@name='floor']")
        assert floor.get("contype") == floor.get("conaffinity") == "0"
    else:
        assert compiler.get("eulerseq") == "XYZ"
    assert root.find("option").get("timestep") == "0.01"
    prefix = "body0_"
    assert root.find("actuator/general").get("name") == prefix + "thruster1"
    assert root.find("actuator/general").get("site") == body.find("site").get("name")


def test_build_does_not_change_pose_or_later_training_build():
    config = scene_config()
    config.Bodies.bodies_list[0].euler = np.array([10.0, 20.0, 30.0])
    before = deepcopy(config.Bodies.bodies_list[0].euler)
    vehicle = load_vehicle("vehicles/astrobee.yaml")
    build_mujoco_xml(config, vehicle, scene="gateway")
    np.testing.assert_array_equal(config.Bodies.bodies_list[0].euler, before)
    root = ET.fromstring(build_mujoco_xml(config, vehicle, scene="training"))
    assert root.find("worldbody/body[@name='body0']").get("euler") == "10.0 20.0 30.0"


def test_planar_configuration_is_rejected():
    config = scene_config()
    config.dof = "3d"
    with pytest.raises(ValueError, match="Only 6DoF"):
        build_mujoco_xml(config, load_vehicle("vehicles/astrobee.yaml"))


def test_dock_sites_and_multiple_bodies_have_distinct_actuators():
    config = scene_config()
    config.gateway = SimpleNamespace(dock_sites=[
        {"name": "dock_a", "pos": [1, 2, 3], "size": 0.1},
        {"name": "dock_b", "pos": [4, 5, 6], "size": [0.2], "quat": [0, 0, 0, 1]},
    ])
    config.Bodies.bodies_list.append(settings({
        "name": "body1", "pos": [4, 5, 6], "euler": [0, 0, 0],
    }))
    vehicle = load_vehicle("vehicles/astrobee.yaml")
    root = ET.fromstring(build_mujoco_xml(config, vehicle))
    dock = root.find("worldbody/body[@name='gateway_full']")
    assert [s.get("name") for s in dock.findall("site")] == ["dock_a", "dock_b"]
    assert dock.find("site[@name='dock_b']").get("quat") == "0 0 0 1"
    names = [act.get("name") for act in root.find("actuator")]
    assert len(names) == len(set(names)) == 2 * len(vehicle.actuators)


def test_generic_vehicle_keeps_geometries_and_omitted_site_default():
    config = scene_config()
    vehicle = load_vehicle("vehicles/cubesat.yaml")
    vehicle = replace(vehicle, actuators=(replace(vehicle.actuators[0], site=None),))
    root = ET.fromstring(build_mujoco_xml(config, vehicle, scene="training"))
    body = root.find("worldbody/body[@name='body0']")
    assert len(body.findall("geom")) == len(vehicle.geoms)
    assert body.find("site").get("name") == "body0_" + vehicle.actuators[0].name


def test_unknown_scene_fails_before_loading_fragments():
    with pytest.raises(ValueError, match="Unknown scene"):
        build_mujoco_xml(scene_config(), load_vehicle("vehicles/astrobee.yaml"), scene="typo")


@pytest.mark.parametrize("scene", ["gateway", "training"])
def test_bundled_scenes_load_outside_checkout(monkeypatch, tmp_path, scene):
    import mujoco

    config = scene_config()
    config.visual.headlight = {key: "0.4 0.4 0.4" for key in ("ambient", "specular", "diffuse")}
    vehicle = load_vehicle("vehicles/astrobee.yaml")
    monkeypatch.chdir(tmp_path)
    model = mujoco.MjModel.from_xml_string(build_mujoco_xml(config, vehicle, scene=scene))
    assert model.nu == len(vehicle.actuators)


@pytest.mark.parametrize("path_kind", ["relative_directory", "absolute_directory", "absolute_file"])
def test_custom_mesh_paths_keep_their_meaning(monkeypatch, tmp_path, path_kind):
    import shutil
    from pathlib import Path
    import mujoco
    from smallsat_sim.model import mujoco_xml

    vehicle = load_vehicle("vehicles/cubesat.yaml")
    mesh = vehicle.geoms[0]
    custom = tmp_path / "custom"
    custom.mkdir()
    source = Path(mujoco_xml.__file__).parent / mesh.mesh
    target = custom / source.name
    shutil.copyfile(source, target)
    config = scene_config()
    config.visual.headlight = {key: "0.4 0.4 0.4" for key in ("ambient", "specular", "diffuse")}
    if path_kind == "absolute_file":
        mesh = replace(mesh, mesh=str(target))
    else:
        config.compiler["meshdir"] = "custom" if path_kind == "relative_directory" else str(custom)
        mesh = replace(mesh, mesh=target.name)
    vehicle = replace(vehicle, geoms=(mesh,))
    monkeypatch.chdir(tmp_path)
    xml = build_mujoco_xml(config, vehicle, scene="training")
    if "meshdir" in config.compiler:
        assert ET.fromstring(xml).find("compiler").get("meshdir") == config.compiler["meshdir"]
    assert mujoco.MjModel.from_xml_string(xml).nmesh == 1
