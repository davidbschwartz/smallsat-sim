"""Resolve fresh AD* defaults with environment-specific endpoints."""

import numpy as np
from smallsat_sim.configuration import load_settings


def planner_settings(*, start=(0, 0, 0), goal=(0, 0, 0)):
    config = load_settings("planners/ad_star.yaml")["3d"]
    config.bounds = np.asarray(config.bounds)
    config.unit_directions = {
        tuple(item.direction): item.cost for item in config.pop("neighbors")
    }
    config.start_pos, config.goal_pos = start, goal
    return config
