"""Regression tests for mission timing and spline geometry."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from smallsat_sim.planners.mission.mission import MissionPlanner, MissionPlannerCubicSpline


@pytest.fixture(scope="module")
def spline():
    return MissionPlannerCubicSpline(SimpleNamespace(using_rl=False), spacing=2.)


def test_dwell_requires_continuous_simulation_time():
    env = SimpleNamespace(using_rl=False, data=SimpleNamespace(time=0.))
    planner = MissionPlanner(env, planner_mode="Waypoint Tracking")
    target = planner.waypoints[0].position.copy()
    with patch("time.time", return_value=1000.), patch("time.monotonic", return_value=1000.):
        planner.get_reference(target)
        env.data.time = 4.
        planner.get_reference(target)
        assert planner.idx_reference_point == 0
        planner.get_reference(target + 100.)
        env.data.time = 6.
        planner.get_reference(target)
        assert planner.idx_reference_point == 0
        env.data.time = 10.9
        planner.get_reference(target)
        assert planner.idx_reference_point == 0
        env.data.time = 11.
        planner.get_reference(target)
        assert planner.idx_reference_point == 1


def test_spline_reference_uses_spline_and_requested_spacing(spline):
    assert spline.spacing == 2.
    expected = spline.get_visualization_points(spacing=2.)[:-1]
    assert len(spline._intermediate_reference) == len(expected)
    spline.idx_reference_point = 10
    position, attitude = spline.get_reference(np.ones(3) * 1000.)
    np.testing.assert_allclose(position[:, 0], expected[10].position)
    np.testing.assert_allclose(attitude[:, 0], expected[10].attitude)


def test_spline_projection_recovers_all_waypoints(spline):
    for point in spline.positions:
        closest, parameter = spline.closest_point_on_trajectory(point)
        np.testing.assert_allclose(closest, point, atol=1e-7)
        np.testing.assert_allclose(
            [spline.x_spline(parameter), spline.y_spline(parameter), spline.z_spline(parameter)],
            closest, atol=1e-7,
        )


def test_spline_projection_beats_dense_sampling(spline):
    parameters = np.linspace(0, spline.total_length, 20000)
    samples = np.stack([spline.x_spline(parameters), spline.y_spline(parameters), spline.z_spline(parameters)], axis=-1)
    for point in ([0., 0., 0.], [-3., -12., 2.], [18., 1., 10.]):
        closest, _ = spline.closest_point_on_trajectory_CS(point)
        assert np.linalg.norm(closest - point) <= np.min(np.linalg.norm(samples - point, axis=1)) + 1e-7


@pytest.mark.parametrize("value", [0., -1., np.nan, np.inf])
@pytest.mark.parametrize("parameter", ["spacing", "clearance_dist"])
def test_mission_rejects_invalid_configuration(parameter, value):
    with pytest.raises(ValueError, match=parameter):
        MissionPlanner(SimpleNamespace(using_rl=False), **{parameter: value})


def test_mission_rejects_unknown_mode():
    with pytest.raises(ValueError, match="planner_mode"):
        MissionPlanner(SimpleNamespace(using_rl=False), planner_mode="typo")


def test_spline_and_waypoints_share_attitude_convention(spline):
    from smallsat_sim.planners.mission.mission import Waypoint
    expected = Waypoint(np.zeros(3), np.array([0., 0., 90.])).attitude
    np.testing.assert_allclose(spline.get_attitude(0), expected)


def test_mission_reset_and_static_overlay():
    from unittest.mock import Mock
    env = SimpleNamespace(using_rl=False, visualization=Mock())
    planner = MissionPlanner(env, planner_mode="Waypoint Tracking")
    count = env.visualization.set_overlay.call_count
    planner.idx_reference_point = 3
    planner.timer_started = True
    planner.reset()
    assert planner.idx_reference_point == 0
    assert not planner.timer_started
    planner.get_reference(np.ones(3) * 100)
    planner.get_reference(np.ones(3) * 100)
    assert env.visualization.set_overlay.call_count == count
    planner.reset(planner.waypoints[5].position)
    assert planner.idx_reference_point == 5


def test_docking_reset():
    from smallsat_sim.planners.mission.docking import DockingPlanner
    planner = DockingPlanner(SimpleNamespace(using_rl=False), np.zeros(3), np.ones(3))
    planner.get_reference(np.r_[np.zeros(3), 1., np.zeros(9)])
    assert planner.stage == 1
    planner.reset()
    assert planner.stage == 0


def make_line(length=1.):
    from smallsat_sim.planners.mission.mission import Line, Waypoint
    return Line(Waypoint(np.zeros(3), np.zeros(3)),
                Waypoint(np.array([length, 0., 0.]), np.array([0., 0., 90.])))


def test_zero_length_line_is_rejected():
    with pytest.raises(ValueError, match="nonzero separation"):
        make_line(0.)


@pytest.mark.parametrize("arc,endpoint", [(-0.0005, "start_point"), (1.0005, "end_point")])
def test_line_clamps_position_and_attitude_together(arc, endpoint):
    line = make_line()
    actual = line.interpolate(arc)
    expected = getattr(line, endpoint)
    np.testing.assert_allclose(actual.position, expected.position)
    np.testing.assert_allclose(actual.attitude, expected.attitude)


@pytest.mark.parametrize("arc", [-0.01, 1.01, np.nan, np.inf])
@pytest.mark.parametrize("mode", ["Linear", "Constant"])
def test_line_rejects_invalid_arc_length(arc, mode):
    with pytest.raises(ValueError, match="arc_length"):
        make_line().interpolate(arc, mode)


@pytest.mark.parametrize("length", [0.1, 0.2, 0.39, 0.4, 0.41])
def test_intermediate_spacing_bounds_every_gap(length):
    from smallsat_sim.planners.mission.mission import Waypoint

    class ShortMission(MissionPlanner):
        def _load_waypoints(self):
            self.waypoints = [Waypoint(np.array([x, 0., 0.]), np.zeros(3))
                              for x in (0., length, 0.)]
            self.segment_types = ["Line", "Line"]

    planner = ShortMission(SimpleNamespace(using_rl=False), spacing=0.2)
    points = np.array([wp.position for wp in planner._intermediate_reference])
    gaps = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    assert np.all(gaps <= 0.2 + 1e-12)
    assert np.all(gaps > 0)
    assert any(np.array_equal(point, [length, 0., 0.]) for point in points)
