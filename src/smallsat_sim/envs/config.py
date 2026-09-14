"""Shared YAML defaults and explicitly seeded initial poses."""

from functools import partial

import numpy as np

from smallsat_sim.planners.ad_star.config import planner_settings

from smallsat_sim.configuration import load_settings, flatten_overrides, apply_overrides, settings


def randomize_initial_pose(body, config, *, rng):
    """Sample the configured pose; keep random draws in Python, not YAML snapshots."""
    position = np.asarray(body.pos, dtype=float) + rng.uniform(
        -config.position_noise, config.position_noise, 3
    )
    attitude = np.array(body.euler, dtype=float, copy=True)
    if "attitude_noise_degrees" in config:
        low, high = config.attitude_noise_degrees
        attitude += rng.randint(low, high, 3)
    return position.tolist(), attitude.tolist()


def _merge_settings(target, values):
    for name, value in values.items():
        if isinstance(value, dict) and isinstance(target.get(name), dict):
            _merge_settings(target[name], value)
        else:
            target[name] = value


def load_env_settings(name):
    """Fresh simulation defaults overlaid by the named environment YAML."""
    config = load_settings("simulation.yaml")
    if name == "cubesat":
        shared = load_settings("environments/astrobee.yaml").control
        _merge_settings(config, settings({"control": {key: shared[key] for key in ("NominalMPCC", "GPMPC")}}))
    _merge_settings(config, load_settings(f"environments/{name}.yaml"))
    return config


_PLANNER_DEFAULTS = {
    "astrobee": ((0, 0, 0), (0, 0, 0)),
}


def resolve_env_config(name, *, overrides=None, seed=None, planner=None):
    """Resolve settings and sample the initial pose before simulator allocation."""
    config = load_env_settings(name)
    if planner is None and name in _PLANNER_DEFAULTS:
        start, goal = _PLANNER_DEFAULTS[name]
        planner = planner_settings(start=start, goal=goal)
    if planner is not None:
        config.planner = planner
    paths = {}
    for path, value in flatten_overrides(overrides or {}).items():
        if path.split(".", 1)[0] in config.control:
            path = f"control.{path}"
        if path in paths:
            raise ValueError(f"Duplicate override path: {path}")
        paths[path] = value
    apply_overrides(config, paths)
    if seed is not None:
        config.sim.seed = seed
    if "position_noise" in config.Bodies:
        rng = np.random.RandomState(config.sim.seed)
        for body in config.Bodies.bodies_list:
            body.pos, body.euler = randomize_initial_pose(body, config.Bodies, rng=rng)
    return config


resolve_astrobee_config = partial(resolve_env_config, "astrobee")
resolve_cubesat_config = partial(resolve_env_config, "cubesat")
