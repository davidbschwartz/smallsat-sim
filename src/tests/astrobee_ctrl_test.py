"""Controller smoke tests against the Astrobee symbolic model."""

import numpy as np
import pytest

from smallsat_sim.model.dynamics import SymbolicModel
from smallsat_sim.envs.config import resolve_astrobee_config
from smallsat_sim.model.vehicle import load_vehicle
from support.standalone_env import StandaloneEnv
from smallsat_sim.planners.mission.mission import MissionPlanner, Waypoint

pytest.importorskip("acados_template", reason="MPCC tests require the optional mpc dependencies")
from smallsat_sim.controllers.mpcc.controller import NominalMPCCController


class ConstantReferencePlanner(MissionPlanner):
    """Simple Planner that returns a constant reference."""

    def _load_waypoints(self) -> None:
        """
        Loads the sparse waypoints that shall be reached
        """
        # Define positional references
        positions = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.01],
        ] * 8

        # Define attitude references (Euler angles)ßßß
        attitudes = [[0.0, 0.0, 0.0]] * 16

        # Define connection type between waypoints
        segment_types = [
            "Line",
        ] * 16

        # Add first point as last point to ensure continuity
        positions.append(positions[0])
        attitudes.append(attitudes[0])

        # Save segment types to self
        self.segment_types = segment_types

        # Create Waypoint objects
        self.waypoints = [
            Waypoint(np.array(pos), np.array(att))
            for pos, att in zip(positions, attitudes)
        ]


def get_observation(noise_level=1.0, *, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    # Set observation (position, orientation, velocity, angular velocity)
    r = np.array([3.47454012, 16.45071431, 0.23199394])
    q = np.array([0.60850556, 0.63437829, -0.46351812, -0.1115174])
    v = np.array([0.0, 0.0, 0.0])
    omega = np.array([0.0, 0.0, 0.0])

    # add noise
    r += rng.random(3) * noise_level
    q += rng.random(4) * noise_level
    q /= np.linalg.norm(q)  # normalize quaternion

    v += rng.random(3) * noise_level
    omega += rng.random(3) * noise_level
    return r, q, v, omega


@pytest.fixture
def env():
    cfg = load_vehicle("vehicles/astrobee.yaml")
    model = SymbolicModel(cfg)
    env = StandaloneEnv(env_cfg=resolve_astrobee_config(), model=model, model_cfg=cfg)
    r, q, v, omega = get_observation(noise_level=0)
    env.set_obs(np.concatenate([r, q]), v, omega)
    return env


@pytest.fixture
def planner(env):
    return MissionPlanner(env)


@pytest.fixture
def const_ref_planner(env):
    return ConstantReferencePlanner(env)


def test_mpcc(env, planner):
    # Define the controller
    ctrl = NominalMPCCController(env, planner)

    rng = np.random.default_rng(7)
    for _ in range(3):
        r, q, v, omega = get_observation(noise_level=0.25, rng=rng)
        env.set_obs(np.concatenate([r, q]), v, omega)
        ctrl_input = ctrl.get_control_input(env)
        assert np.isfinite(ctrl_input).all()


def test_zero_input_mpcc(env, const_ref_planner):
    ctrl = NominalMPCCController(env, const_ref_planner)
    pose = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    env.set_obs(pose, np.zeros(3), np.zeros(3))
    ctrl_input = ctrl.get_control_input(env)

    # Forward propagate model
    x_dot = env.symbolic_model.f_expl_expr_func(
        np.concatenate([pose, np.zeros(6)]), ctrl_input
    ).toarray()
    velocities = x_dot[7:]
    assert np.abs(velocities).max() < 0.06
