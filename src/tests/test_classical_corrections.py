"""Physical controller and planner regressions from the classical assessment."""
from types import SimpleNamespace as NS
from unittest.mock import Mock

import casadi as ca
import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from smallsat_sim.configuration import load_settings
from smallsat_sim.envs.config import load_env_settings
from smallsat_sim.model.vehicle import load_vehicle
from smallsat_sim.model.dynamics import SymbolicModel
from smallsat_sim.controllers.lqr.controller import LQRController
from smallsat_sim.controllers.pd.common import compute_control, allocate_wrench
from smallsat_sim.controllers.state import body_state
from smallsat_sim.controllers.nominal_mpc.reference import attitude_cost, MPCSolverError
from smallsat_sim.planners.ad_star.ad_star import ADStarPlanner, NoPathError
from smallsat_sim.planners.ad_star.config import planner_settings
from smallsat_sim.planners.clearance import ClearanceScene
from smallsat_sim.planners.mission.docking import DockingPlanner


def control_env(vehicle, state, frame="body"):
    asset = load_vehicle("examples/demo_spacecraft.yaml" if vehicle == "demo" else f"vehicles/{vehicle}.yaml")
    cfg = load_env_settings("cubesat" if vehicle in ("demo", "cubesat") else "astrobee")
    cfg.sim.obs.v_frame = frame
    model = SymbolicModel(asset)
    return NS(model_cfg=asset, symbolic_model=model, env_cfg=cfg, get_obs=lambda: state,
              obs=state, args=NS(video=False), run_id=0, data=NS(time=0.), visualization=None)


@pytest.mark.parametrize("vehicle", ["astrobee", "cubesat", "sprint", "demo"])
@pytest.mark.parametrize("target", [[0, 0, 0], [35, -40, 120], [180, 0, 0]])
def test_lqr_opposes_attitude_error_for_every_asset_and_rotated_target(vehicle, target):
    desired = Rotation.from_euler("xyz", target, degrees=True)
    actual = desired * Rotation.from_rotvec([.1, 0., 0.])
    obs = np.r_[np.zeros(3), actual.as_quat(scalar_first=True), np.zeros(6)]
    env = control_env(vehicle, obs)
    planner = NS(get_reference=lambda obs: (np.zeros(3), desired.as_quat(scalar_first=True)))
    controller = LQRController(env, planner)
    command = controller.get_control_input(env)
    wrench = env.symbolic_model.mixer @ command
    assert wrench[3] < 0
    assert np.all(command >= 0)
    assert np.all(command <= controller._upper)
    obs[3:7] *= -1
    np.testing.assert_allclose(controller.get_control_input(env), command, atol=1e-9)


def test_lqr_velocity_frames_give_identical_damping():
    q = Rotation.from_euler("xyz", [35, -40, 120], degrees=True)
    ref = q.as_quat(scalar_first=True)
    state = np.r_[np.zeros(3), ref, [.2, -.1, .05], np.zeros(3)]
    planner = NS(get_reference=lambda obs: (np.zeros(3), ref))
    body = control_env("astrobee", state.copy())
    inertial_state = state.copy()
    inertial_state[7:10] = q.apply(state[7:10])
    inertial = control_env("astrobee", inertial_state, "inertial")
    a, b = LQRController(body, planner), LQRController(inertial, planner)
    np.testing.assert_allclose(a.get_control_input(body), b.get_control_input(inertial), atol=1e-9)
    assert np.dot((body.symbolic_model.mixer @ a.get_control_input(body))[:3], state[7:10]) < 0


def test_pd_noncommuting_attitude_error_is_body_torque():
    desired = Rotation.from_euler("xyz", [30, -45, 80], degrees=True)
    actual = desired * Rotation.from_rotvec([.1, 0., 0.])
    obs = np.r_[np.zeros(3), actual.as_quat(scalar_first=True), np.zeros(6)][None]
    reference = np.r_[np.zeros(3), desired.as_quat(scalar_first=True)][None]
    # Identity wrench mapping allows an independent torque expectation.
    command = compute_control(obs, reference, np.zeros(3), (1., 1., 1., 1.),
                              np.eye(6), (), velocity_frame="body", xp=np)
    np.testing.assert_allclose(command[0, 3:], [-np.sin(.05), 0., 0.], atol=1e-12)


def test_generic_allocation_respects_bounds_and_recovers_feasible_wrench():
    # Oblique, unbalanced columns: group-offset shifting cannot preserve wrench.
    rng = np.random.default_rng(45)
    matrix = rng.normal(size=(6, 15))
    feasible = np.linspace(.02, .3, 15)
    wrench = (matrix @ feasible)[None]
    raw = wrench @ np.linalg.pinv(matrix).T
    bounds = (np.zeros(15), np.full(15, .4))
    result = allocate_wrench(wrench, raw, matrix, (np.arange(15),), bounds, np)
    assert np.all(result >= 0) and np.all(result <= .4)
    np.testing.assert_allclose(result @ matrix.T, wrench, atol=1e-3)
    import jax.numpy as jnp
    np.testing.assert_allclose(allocate_wrench(jnp.asarray(wrench), jnp.asarray(raw), matrix,
                                              (np.arange(15),), bounds, jnp), result, atol=2e-6)


def test_body_state_converts_velocity_without_mutating_observation():
    q = Rotation.from_euler("z", 90, degrees=True).as_quat(scalar_first=True)
    obs = np.r_[np.zeros(3), q, [1., 0., 0.], np.zeros(3)]
    original = obs.copy()
    state = body_state(control_env("cubesat", obs, "inertial"))
    np.testing.assert_allclose(state[7:10], [0., -1., 0.], atol=1e-12)
    np.testing.assert_array_equal(obs, original)


def test_contouring_attitude_cost_is_sign_invariant():
    q, desired = ca.SX.sym("q", 4), ca.SX.sym("d", 4)
    cost = ca.Function("cost", [q, desired], [attitude_cost(q, desired, np.eye(4))])
    target = Rotation.from_euler("xyz", [20, 35, 45], degrees=True).as_quat(scalar_first=True)
    actual = Rotation.from_euler("xyz", [-10, 60, 100], degrees=True).as_quat(scalar_first=True)
    assert float(cost(target, -target)) == pytest.approx(0, abs=1e-12)
    assert float(cost(actual, target)) == pytest.approx(float(cost(-actual, target)))
    assert float(cost(actual, target)) == pytest.approx(float(cost(actual, -target)))


def scene_env(obstacle='<geom type="box" pos="1 0 0" size=".1 .3 .3"/>'):
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><option gravity="0 0 0"/>
      <worldbody>{obstacle}<body name="body0"><freejoint/>
      <geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cfg = planner_settings(start=(0, 0, 0), goal=(2, 0, 0))
    cfg.bounds = np.array([[0, -1, 0], [2, 1, 0]])
    return NS(model=model, data=data, using_rl=False, env_cfg=NS(planner=cfg))


def test_adstar_plans_around_obstacle_and_holds_goal():
    env = scene_env()
    original = env.data.qpos.copy()
    planner = ADStarPlanner(env)
    assert planner.path[0] == (0., 0., 0.) and planner.path[-1] == (2., 0., 0.)
    assert len(planner.path) >= 3
    assert all(planner.scene.segment_is_clear(a, b, clearance=planner.clearance)
               for a, b in zip(planner.path[:-1], planner.path[1:]))
    np.testing.assert_array_equal(env.data.qpos, original)
    planner.idx_reference_point = len(planner.path) - 1
    position, quaternion = planner.get_reference(np.r_[2., 0., 0., 1., np.zeros(9)])
    np.testing.assert_allclose(position[:, 0], [2., 0., 0.])
    assert quaternion.shape == (4, 1)
    closest, arc = planner.closest_point_on_trajectory([2., 0., 0.])
    np.testing.assert_allclose(closest, [2., 0., 0.], atol=1e-12)
    assert arc > 0


def test_adstar_no_path_and_cancelled_thread():
    env = scene_env('<geom type="box" pos="1 0 0" size=".1 2 1"/>')
    with pytest.raises(NoPathError):
        ADStarPlanner(env)
    planner = ADStarPlanner(scene_env(''))
    planner.start_planning_thread()
    planner.stop_planning_thread()
    assert planner.planning_thread is None


def test_sweep_catches_thin_diagonal_obstacle_and_vehicle_width():
    # Beyond one grid-resolution unit on a diagonal; old point-ray test missed it.
    env = scene_env('<geom type="box" pos=".85 .85 0" size=".01 .01 .2"/>')
    scene = ClearanceScene(env)
    assert not scene.segment_is_clear([0., 0., 0.], [1., 1., 0.])
    # Centerline misses, spacecraft surface clips.
    env = scene_env('<geom type="box" pos=".5 .09 0" size=".01 .01 .2"/>')
    assert not ClearanceScene(env).segment_is_clear([0., 0., 0.], [1., 0., 0.])


def test_docking_requires_alignment_low_rates_and_contact():
    planner = DockingPlanner(NS(using_rl=False), np.zeros(3), np.array([1., 0, 0]))
    obs = np.r_[np.zeros(3), 1., np.zeros(9)]
    obs[7] = .2
    planner.get_reference(obs)
    assert planner.stage == 0
    obs[7] = 0.
    obs[3:7] = Rotation.from_euler("z", 90, degrees=True).as_quat(scalar_first=True)
    planner.get_reference(obs)
    assert planner.stage == 0
    obs[3:7] = [1., 0., 0., 0.]
    planner.get_reference(obs)
    assert planner.stage == 1
    obs[0] = 1.
    assert not planner.is_docked(obs)  # no contact observation
    planner.require_contact = False
    assert planner.is_docked(obs)
    obs[10] = .2
    assert not planner.is_docked(obs)


def test_docking_rejects_invalid_approach():
    with pytest.raises(ValueError, match="intersects"):
        DockingPlanner(scene_env(), np.zeros(3), np.array([2., 0., 0.]))


@pytest.mark.parametrize("module,classname", [("nominal_mpc", "NominalMPCController"), ("mpcc", "NominalMPCCController")])
def test_mpc_rejects_new_failed_solve_before_applying_or_saving(module, classname):
    pytest.importorskip("acados_template")
    from importlib import import_module
    cls = getattr(import_module(f"smallsat_sim.controllers.{module}.controller"), classname)
    ctrl = cls.__new__(cls)
    env = control_env("cubesat", np.r_[np.zeros(3), 1., np.zeros(9)])
    ctrl.ocp_solver = Mock(status=0)
    def solve(*args, **kwargs):
        ctrl.ocp_solver.status = 4
        return np.zeros(12 if module == "nominal_mpc" else 13)
    ctrl.ocp_solver.solve_for_x0.side_effect = solve
    ctrl.ctrl_cfg = NS(N=2)
    ctrl.planner = NS(get_reference=lambda obs: (np.zeros((3, 1)), np.array([[1.], [0.], [0.], [0.]])))
    ctrl._set_params = Mock()
    ctrl.theta_prev = [0., 0., 0.]
    ctrl._visualize_prediction = Mock()
    with pytest.raises(MPCSolverError):
        ctrl.get_control_input(env)
    ctrl._visualize_prediction.assert_not_called()
    assert not hasattr(ctrl, "u_past")


def test_cube_defaults_penalize_pose_and_all_classical_names_registered():
    cfg = load_env_settings("cubesat")
    assert np.all(np.diag(cfg.control.NominalMPC.cost.Q)[:7] > 0)
    assert np.all(np.diag(cfg.control.NominalMPC.cost.T)[:7] > 0)
    from smallsat_sim.api.experiments import list_controllers, list_planners
    assert {"pd", "lqr", "mpc", "mpcc", "gp_mpc"} <= set(list_controllers())
    assert {"oracle", "mission", "mission_spline", "docking", "ad_star"} <= set(list_planners())


def test_contouring_line_origin_tracks_unwrapped_laps():
    from smallsat_sim.planners.mission.mission import MissionPlanner
    planner = MissionPlanner(NS(using_rl=False))
    path = planner.trajectory
    for s in (.01, path.length / 2, path.length + .01, 3 * path.length + .01):
        origin = path._get_start_point_segment(s)
        start = path._get_start_arc_length_segment(s)
        tangent = path._get_tangent_segment(s)
        np.testing.assert_allclose(origin + (s - start) * tangent,
                                   path.get_intermediate_reference(s).position, atol=1e-10)


def test_unsupported_contouring_pair_fails_before_solver_allocation():
    from smallsat_sim.planners.validation import require_contouring_planner
    from smallsat_sim.planners.oracle.oracle import OraclePlanner
    planner = OraclePlanner(NS(using_rl=False))
    with pytest.raises(ValueError, match="piecewise-line"):
        require_contouring_planner(planner)


@pytest.mark.parametrize("learning_enabled", [False, True])
def test_gp_failure_rejects_solution_and_reset_discards_learning(learning_enabled):
    pytest.importorskip("l4acados")
    from smallsat_sim.controllers.gp_mpc.controller import GPMPC
    ctrl = GPMPC.__new__(GPMPC)
    env = control_env("astrobee", np.r_[np.zeros(3), 1., np.zeros(9)])
    ctrl._previous_time = -.05
    ctrl._compute_residual = Mock(return_value=None)
    ctrl.has_logger = False
    ctrl.ctrl_cfg = NS(N=1, Ts=.05, learning_enabled=learning_enabled)
    ctrl.N, ctrl.nu = 1, 13
    ctrl.last_solution = {'states':np.zeros((2, 14)), 'inputs':np.zeros((1, 13))}
    ctrl.gp_mpc = Mock(ocp_solver=Mock(status=0))
    ctrl.gp_mpc.solve.return_value = 4
    ctrl.theta_prev = [0., 0.]
    ctrl.planner = NS(closest_point_on_trajectory=lambda obs: (np.zeros(3), 0.), trajectory=NS(length=2.))
    ctrl._set_params = Mock()
    ctrl._visualize_prediction = Mock()
    with pytest.raises(MPCSolverError):
        ctrl.get_control_input(env)
    assert ctrl._compute_residual.call_count == int(learning_enabled)
    ctrl.gp_mpc.get_solution.assert_not_called()
    ctrl._visualize_prediction.assert_not_called()
    ctrl._fresh_residual_model = NS(data=[])
    ctrl.gp_mpc.residual_model = NS(data=[1, 2])
    ctrl._initialize_solver = Mock()
    ctrl.reset(env)
    ctrl.gp_mpc.ocp_solver.reset.assert_called_once()
    assert ctrl.gp_mpc.residual_model.data == []
    assert ctrl.gp_mpc.residual_model is not ctrl._fresh_residual_model
    assert ctrl._previous_time is None
    assert not np.any(ctrl.u_past)


def test_gp_feature_and_window_adapters_use_current_dependency_contract():
    pytest.importorskip("l4acados")
    import torch
    from smallsat_sim.controllers.gp_mpc.online_learning.utils import ScaleFeatureSelector
    from smallsat_sim.controllers.gp_mpc.online_learning.strategies import SlidingWindowPlus
    selector = ScaleFeatureSelector(np.array([0., 1., 1.]))
    actual = selector(torch.tensor([[4., 5., 6.]], dtype=torch.float64))
    np.testing.assert_allclose(actual, [[5., 6.]])
    strategy = SlidingWindowPlus(max_num_points=8, device="cpu")
    assert strategy.device == 'cpu' and strategy.use_newest
    assert strategy.flag_counter == 0


def test_adstar_worker_does_not_erase_cancellation_requested_before_search():
    import threading
    planner = ADStarPlanner(scene_env(''))
    real_replan = planner.replan
    entered, release = threading.Event(), threading.Event()
    def delayed():
        entered.set()
        assert release.wait(timeout=2)
        real_replan()
    planner.replan = delayed
    planner.start_planning_thread()
    assert entered.wait(timeout=2)
    planner._stop.set()
    release.set()
    planner.stop_planning_thread()
    assert isinstance(planner._error, NoPathError)
    assert planner.path == []


def test_classical_factory_builds_lqr_and_tunes_custom_pd_without_hiding_overrides():
    from smallsat_sim.api.experiments import make_experiment
    for options in ({'controller':'lqr'}, {'controller':'pd', 'asset':'examples/demo_spacecraft.yaml',
                                         'overrides':{'PD.gains.Kp_x':.7}}):
        experiment = make_experiment(vehicle='cubesat', planner_radius=0., log=False, **options)
        try:
            action = experiment.controller.get_control_input(experiment.env)
            assert np.isfinite(action).all() and action.shape == (12,)
            if options['controller'] == 'pd':
                assert experiment.controller.Kp_x == .7
                assert experiment.controller.Kd_x == 2 * 4. * .35
        finally:
            experiment.env.close()


def test_docking_rejects_initial_penetration_even_with_clear_final_approach():
    env = scene_env('<geom type="box" pos="0 0 0" size=".1 .1 .1"/>')
    planner = DockingPlanner(env, np.array([2., 0., 0.]), np.array([3., 0., 0.]))
    with pytest.raises(ValueError, match="Initial docking pose"):
        planner.get_reference(np.r_[np.zeros(3), 1., np.zeros(9)])


@pytest.mark.parametrize("frame", ["body", "inertial"])
def test_live_classical_observation_uses_named_free_joint_and_current_state(frame):
    from smallsat_sim.envs.base_env import BaseEnv
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="body0"><freejoint/><geom size=".1" mass="1"/>
        <body pos="5 0 0"><geom size=".1" mass=".1"/></body>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    quaternion = Rotation.from_euler("z", 90, degrees=True).as_quat(scalar_first=True)
    data.qpos[:] = np.r_[1., 2., 3., quaternion]
    data.qvel[:] = [1., 0., 0., .1, .2, .3]
    env = BaseEnv.__new__(BaseEnv)
    env.model, env.data = model, data
    env.env_cfg = NS(sim=NS(noise=NS(add_obs_noise=False)))
    env.set_obs(v_frame=frame)
    expected_velocity = [0., -1., 0.] if frame == "body" else [1., 0., 0.]
    np.testing.assert_allclose(env.get_obs(), np.r_[1., 2., 3., quaternion, expected_velocity, .1, .2, .3], atol=1e-12)
