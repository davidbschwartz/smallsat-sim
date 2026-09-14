"""Build MuJoCo XML from physical vehicles and explicit scene choices.

Scene fragments contain static assets/geometry. Bodies, joints, inertias, sites,
actuators and configurable docking sites are assembled here, once at construction.
"""

from copy import deepcopy
from pathlib import Path
from typing import Literal
from xml.etree import ElementTree as ET

from smallsat_sim.model.vehicle import VehicleSpec

Scene = Literal["gateway", "training"]
_MODEL_DIR = Path(__file__).resolve().parent
_FRAGMENT_DIR = _MODEL_DIR / "xml"


def _xml_value(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    return " ".join(str(item) for item in value)


def _add_model_settings(root, config, scene):
    compiler = {
        "texturedir": str(_MODEL_DIR),
        "meshdir": str(_MODEL_DIR),
    }
    # Training historically uses MuJoCo's default Euler convention.
    if scene != "training":
        compiler["eulerseq"] = "XYZ"
    compiler.update({
        key: str(value) for key, value in (getattr(config, "compiler", {}) or {}).items()
        if key != "convexhull"  # Removed in MuJoCo 3.2.7.
    })
    ET.SubElement(root, "compiler", compiler)
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "headlight", {
        key: str(config.visual["headlight"][key])
        for key in ("ambient", "specular", "diffuse")
    })
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth=str(config.renderer.width),
                  offheight=str(config.renderer.height))
    options = {"gravity": "0 0 0", "timestep": str(config.sim.dt)}
    ET.SubElement(root, "option", options)
    defaults = ET.SubElement(root, "default")
    visual = ET.SubElement(defaults, "default", {"class": "visual"})
    ET.SubElement(visual, "geom", group="2", type="mesh", contype="0", conaffinity="0")
    collision = ET.SubElement(defaults, "default", {"class": "collision"})
    ET.SubElement(collision, "geom", group="3", type="mesh")


def _add_dock_sites(gateway_body, config):
    gateway = getattr(config, "gateway", None)
    for site in getattr(gateway, "dock_sites", []):
        ET.SubElement(gateway_body, "site", {
            "name": site["name"], "type": "sphere", "pos": _xml_value(site["pos"]),
            "quat": _xml_value(site.get("quat", [1, 0, 0, 0])),
            "size": _xml_value(site.get("size", [0.2])),
            "rgba": _xml_value(site.get("rgba", [1.0, 0.3, 0.1, 0.8])),
        })


def _add_vehicle_meshes(assets, vehicle):
    names = set()
    for geom in vehicle.geoms:
        if geom.type == "mesh" and geom.name not in names:
            names.add(geom.name)
            ET.SubElement(assets, "mesh", name=geom.name, file=geom.mesh,
                          scale=_xml_value(geom.asset_scale))


def _add_vehicle_geoms(body, vehicle):
    for geom in vehicle.geoms:
        attributes = {"type": geom.type}
        if geom.type == "mesh":
            attributes["mesh"] = geom.name
        else:
            attributes["size"] = _xml_value(geom.size)
        attributes.update(pos=_xml_value(geom.pos), euler=_xml_value(geom.euler))
        ET.SubElement(body, "geom", attributes)


def build_mujoco_xml(
    config,
    vehicle: VehicleSpec,
    *,
    scene: Scene = "gateway",
) -> str:
    """Build a scene using environment-owned body poses and physical vehicle data.

    Both scenes use free joints; training has a non-colliding floor.
    """
    if getattr(config, "dof", "6d") != "6d":
        raise ValueError("Only 6DoF simulation is supported; remove the planar dof setting.")
    if scene not in ("gateway", "training"):
        raise ValueError(f"Unknown scene: {scene}")
    scene_xml = ET.parse(_FRAGMENT_DIR / f"{scene}.xml").getroot()
    root = ET.Element("mujoco", model=config.model)
    _add_model_settings(root, config, scene)
    assets = ET.SubElement(root, "asset")
    assets.extend(scene_xml.find("asset"))

    use_astrobee_shell = vehicle.assets.kind == "astrobee"
    shell = None
    if use_astrobee_shell:
        shell = ET.parse(_FRAGMENT_DIR / "astrobee.xml").getroot()
        assets.extend(shell.find("asset"))
    else:
        _add_vehicle_meshes(assets, vehicle)

    world = ET.SubElement(root, "worldbody")
    world.extend(scene_xml.find("worldbody"))
    if scene == "gateway":
        _add_dock_sites(world.find("body[@name='gateway_full']"), config)

    actuators = ET.SubElement(root, "actuator")
    for pose in config.Bodies.bodies_list:
        euler = list(pose.euler)
        body = ET.SubElement(world, "body", name=pose.name,
                             pos=_xml_value(pose.pos), euler=_xml_value(euler))
        ET.SubElement(body, "freejoint")
        physical = vehicle.physical
        ET.SubElement(body, "inertial", pos=_xml_value(physical.com_offset),
                      mass=str(physical.mass), diaginertia=_xml_value(physical.diag_inertia))
        if shell is not None:
            body.extend(deepcopy(list(shell.find("worldbody"))))
        else:
            _add_vehicle_geoms(body, vehicle)
        prefix = f"{pose.name}_"
        for thruster in vehicle.actuators:
            site_name = prefix + (thruster.site or thruster.name)
            ET.SubElement(body, "site", name=site_name, pos=_xml_value(thruster.pos),
                          size=str(thruster.size))
            ET.SubElement(actuators, "general", name=prefix + thruster.name,
                          site=site_name, gear=_xml_value(thruster.gear),
                          forcerange=_xml_value(thruster.forcerange),
                          ctrlrange=_xml_value(thruster.ctrlrange),
                          forcelimited=thruster.forcelimited)
    ET.indent(root)
    return ET.tostring(root, encoding="unicode")
