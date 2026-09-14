"""Check the packaged nominal model against analytic force/torque responses."""
import mujoco
import numpy as np

from smallsat_sim.envs.config import resolve_astrobee_config
from smallsat_sim.model.mujoco_xml import build_mujoco_xml
from smallsat_sim.model.vehicle import load_vehicle


def test_sprint_pd_defaults_and_explicit_overrides():
    from smallsat_sim.api.experiments import ExperimentSpec, get_env
    for name in ('sprint', 'sprint_rl'):
        builder = get_env(name).config_builder
        cfg = builder(ExperimentSpec(vehicle=name)).env
        if name == 'sprint_rl':
            assert cfg.control.PD.gains.Kd_q == .2
            assert cfg.control.PD.gains.Kp_x == .4
        else:
            vehicle = load_vehicle('vehicles/sprint.yaml')
            assert cfg.control.PD.gains.Kd_q == 2 * max(vehicle.physical.diag_inertia) * .8
            assert cfg.control.PD.gains.Kp_x == vehicle.physical.mass * .35**2
        for path in ('PD.gains.Kd_q', 'control.PD.gains.Kd_q'):
            cfg = builder(ExperimentSpec(vehicle=name, overrides={path: .3})).env
            assert cfg.control.PD.gains.Kd_q == .3


def test_sprint_signed_wrenches_and_saturated_mujoco_response():
    vehicle = load_vehicle('vehicles/sprint.yaml')
    cfg = resolve_astrobee_config()
    model = mujoco.MjModel.from_xml_string(build_mujoco_xml(cfg, vehicle, scene='training'))
    body = model.body('body0').id
    B = np.array([np.r_[a.gear, np.cross(a.pos, a.gear)] for a in vehicle.actuators]).T
    assert np.linalg.matrix_rank(B) == 6
    np.testing.assert_allclose(B @ np.ones(12), 0, atol=1e-14)
    force = .085 * 4.4482216152605
    arm = .1016
    inertia = 464 * .45359237 * .0254**2
    np.testing.assert_allclose(model.body_inertia[body], inertia)
    np.testing.assert_allclose(model.body_mass[body], 35 * .45359237)
    assert np.all(model.actuator_forcelimited)
    for axis in range(6):
        for sign in [-1, 1]:
            target = np.zeros(6)
            target[axis] = sign * (2 * force if axis < 3 else 2 * force * arm)
            u = np.linalg.pinv(B) @ target
            # Each four-jet group has a null sum; offset that group to positive forces.
            for start in (0, 4, 8):
                u[start:start + 4] -= min(0, u[start:start + 4].min())
            assert u.min() >= -1e-12 and u.max() <= force + 1e-12
            np.testing.assert_allclose(B @ u, target, atol=1e-12)
            data = mujoco.MjData(model)
            data.qpos[:3] = [0, 0, 2]
            data.qpos[3:7] = [1, 0, 0, 0]
            data.ctrl[:] = u
            mujoco.mj_forward(model, data)
            expected = target / np.r_[np.full(3, vehicle.physical.mass), np.full(3, inertia)]
            np.testing.assert_allclose(data.qacc, expected, atol=1e-10)
    data.ctrl[:] = 10
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.actuator_force, force)
    data.ctrl[:] = -10
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.actuator_force, 0)
