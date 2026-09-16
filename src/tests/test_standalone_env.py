"""The simulator-free test adapter satisfies the mission planner interface."""

import numpy as np
import pytest

from support.standalone_env import StandaloneEnv
from smallsat_sim.envs.config import resolve_astrobee_config as resolve_config
from smallsat_sim.planners.mission.mission import MissionPlanner


def test_standalone_observations_support_mission_planning():
    env = StandaloneEnv(resolve_config(), model=None, model_cfg=None)
    with pytest.raises(RuntimeError, match="set_obs"):
        env.get_obs()
    pose = np.array([0., 0., 0., 1., 0., 0., 0.])
    env.set_obs(pose, np.zeros(3), np.zeros(3))
    obs = env.get_obs()
    obs[0] = 100.
    assert env.obs[0] == 0.
    planner = MissionPlanner(env)
    point, distance = planner.closest_point_on_trajectory(env.obs[:3])
    assert np.isfinite(point).all() and np.isfinite(distance)
