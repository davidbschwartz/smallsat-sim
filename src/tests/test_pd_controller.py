"""Physical-state parity and allocation regression checks for both PD adapters."""
from types import SimpleNamespace as NS

import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from smallsat_sim.controllers.pd.controller import PDController
from smallsat_sim.controllers.pd.vectorized_controller import VectorizedPDController


@pytest.fixture
def model():
    sites, actuators = [], []
    for axis in range(3):
        for sign in (-1, 1):
            for arm in (-0.1, 0.1):
                index = len(sites)
                pos, gear = np.zeros(3), np.zeros(6)
                pos[(axis + 1) % 3] = arm
                gear[axis] = sign
                sites.append(f'<site name="s{index}" pos="{" ".join(map(str, pos))}"/>')
                actuators.append(f'<general site="s{index}" gear="{" ".join(map(str, gear))}"/>')
    # Interleave force axes and put a marker site before all actuator sites.
    order = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
    return mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint/><geom size="0.1"/>'
        '<site name="marker" pos="9 8 7"/>' + ''.join(sites) +
        '</body></worldbody><actuator>' + ''.join(actuators[i] for i in order) +
        '</actuator></mujoco>'
    )


def env_for(model, obs, frame="body"):
    return NS(
        model=model, num_envs=1 if obs.ndim == 1 else len(obs),
        get_obs=lambda: obs,
        env_cfg=NS(
            control=NS(PD=NS(control_decimation=1, gains=NS(Kp_x=0.2, Kd_x=1., Kp_q=3., Kd_q=5.))),
            sim=NS(obs=NS(v_frame=frame)),
            Bodies=NS(bodies_list=[NS(pos=[0., 0., 0.])]),
        ),
    )


@pytest.mark.parametrize("frame", ["body", "inertial"])
def test_single_and_batched_pd_match_physical_wrench(model, frame):
    rot = Rotation.from_euler("xyz", [30, -25, 80], degrees=True)
    quat = rot.as_quat(scalar_first=True)
    position, velocity, omega = np.array([.3, -.2, .1]), np.array([.2, -.1, .05]), np.array([.01, -.02, .03])
    reference = np.array([.1, .2, -.1, 1., 0., 0., 0.])
    body_obs = np.concatenate((position, quat, rot.inv().apply(velocity), omega))
    single_obs = body_obs.copy()
    if frame == "inertial":
        single_obs[7:10] = velocity
    planner = NS(get_reference=lambda obs: (reference[:3, None], reference[3:, None]))
    single_env = env_for(model, single_obs, frame)
    batch_env = env_for(model, jnp.asarray(np.stack((body_obs, body_obs))))
    single = PDController(single_env, planner)
    batch = VectorizedPDController(batch_env, planner)
    u = single.get_control_input(single_env)
    u_batch = np.asarray(batch.get_control_input(batch_env, jnp.asarray(reference)))
    assert u.shape == (12,)
    assert u_batch.shape == (2, 12)
    assert np.all(u >= 0) and np.all(u_batch >= 0)
    np.testing.assert_allclose(u_batch, np.stack((u, u)), atol=2e-6, rtol=2e-6)
    # Independent force/torque expectation: allocation must preserve the PD wrench.
    force = .2 * rot.inv().apply(reference[:3] - position) - rot.inv().apply(velocity)
    torque = -3. * quat[1:] - 5. * omega
    np.testing.assert_allclose(single.B_matrix @ u, np.r_[force, torque], atol=1e-12)
    one_env = env_for(model, jnp.asarray(body_obs[None, :]))
    np.testing.assert_allclose(batch.get_control_input(one_env, jnp.asarray(reference))[0], u, atol=2e-6)


def test_mixer_resolves_actuator_sites_and_constraints_preserve_wrench(model):
    env = env_for(model, np.r_[np.zeros(3), 1., np.zeros(9)])
    controller = PDController(env, None)
    for i in range(model.nu):
        actuator = model.actuator(i)
        site = model.site(int(actuator.trnid[0]))
        expected = actuator.gear.copy()
        expected[3:] += np.cross(site.pos, actuator.gear[:3])
        np.testing.assert_allclose(controller.B_matrix[:, i], expected)
    raw = np.linspace(-1, 1, model.nu)
    constrained = controller._apply_ctrl_constraint(env, raw.copy())
    assert np.all(constrained >= 0)
    np.testing.assert_allclose(controller.B_matrix @ constrained, controller.B_matrix @ raw, atol=1e-12)


def test_vectorized_default_target_and_position_only_reference(model):
    obs = jnp.array([[0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    env = env_for(model, obs)
    controller = VectorizedPDController(env, None)
    np.testing.assert_allclose(controller.get_control_input(env), 0., atol=1e-7)
    position = jnp.array([[.1, .2, .3]])
    full = jnp.concatenate((position, obs[:, 3:7]), axis=1)
    np.testing.assert_allclose(controller.get_control_input(env, position), controller.get_control_input(env, full))
