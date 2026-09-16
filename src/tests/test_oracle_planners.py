"""Geometry and interface regressions for the two oracle adapters."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from smallsat_sim.planners.oracle.geometry import closest_point_on_reference, circular_reference
from smallsat_sim.planners.oracle.oracle import OraclePlanner
from smallsat_sim.planners.oracle.oracle_rl import OraclePlannerRL


def env():
    return SimpleNamespace(viewer=None, renderer=None, using_rl=False, num_envs=2)


@pytest.mark.parametrize('plane', ['xy', 'yz', 'xz'])
def test_circle_preserves_geometry(plane):
    radius, spacing = 3.0, 1.0
    offsets = np.array([1., 2., 10.17])
    angles = np.arange(int(2 * np.pi * radius // spacing)) * spacing / radius
    c, s, z = radius * np.cos(angles), radius * np.sin(angles), np.zeros_like(angles)
    expected = np.stack({'xy': (c, s, z), 'yz': (z, s, c), 'xz': (c, z, s)}[plane], axis=1) + offsets
    for xp in (np, jnp):
        np.testing.assert_allclose(circular_reference(radius, spacing, plane, offsets, xp=xp), expected, atol=2e-6)


def test_adapter_outputs_and_independent_progress():
    single = OraclePlanner(env(), radius=3., spacing=1., plane='xy', z_offset=10.17)
    batched = OraclePlannerRL(env())
    obs = np.array([single.reference_points[0], [100., 100., 100.]])
    position, quaternion = single.get_reference(obs[0])
    reference = batched.get_reference(jnp.asarray(obs))
    assert position.shape == (3, 1)
    assert quaternion.shape == (4, 1)
    assert reference.shape == (2, 7)
    np.testing.assert_allclose(reference[0, :3], position[:, 0], atol=1e-6)
    np.testing.assert_array_equal(batched.reference_point_indices, [1, 0])
    np.testing.assert_array_equal(reference[:, 3:], [[1, 0, 0, 0]] * 2)
    # Completing one environment must not mark the other complete.
    batched.reference_point_indices = jnp.array([len(batched.reference_points) - 1, 0])
    batched.get_reference(jnp.stack([batched.reference_points[-1], jnp.asarray(obs[1])]))
    np.testing.assert_array_equal(batched.completed_path, [False, False])
    batched.get_reference(jnp.asarray(obs))
    np.testing.assert_array_equal(batched.completed_path, [True, False])


def test_projection_on_segments_and_closed_seam():
    points = np.array([[0., 0., 0.], [2., 0., 0.], [2., 2., 0.], [0., 2., 0.]])
    queries = np.array([[0.5, -1., 0.], [-1., 0.5, 0.], [3., 1.5, 0.]])
    expected = [[0.5, 0., 0.], [0., 0.5, 0.], [2., 1.5, 0.]]
    for xp in (np, jnp):
        project = lambda q: closest_point_on_reference(xp.asarray(points), q, 2., xp=xp)
        if xp is jnp:
            project = jax.jit(project)
        positions, arcs = project(xp.asarray(queries))
        np.testing.assert_allclose(positions, expected)
        np.testing.assert_allclose(arcs, [0.5, 7.5, 3.5])


def test_single_point_targets_and_projection_contracts():
    single = OraclePlanner(env(), radius=0.)
    batched = OraclePlannerRL(env(), radius=0.)
    with np.errstate(divide='raise', invalid='raise'):
        position, arc = single.closest_point_on_trajectory(np.ones(3))
    np.testing.assert_array_equal(position, [0., 0., 0.])
    assert arc == 0.
    positions = batched.closest_point_on_trajectory(jnp.ones((2, 3)))
    assert positions.shape == (2, 3)
    np.testing.assert_allclose(positions, [[0., 0., 10.17]] * 2)


@pytest.mark.parametrize('radius,spacing', [(1., 0.), (1., -1.), (0.1, 10.)])
def test_invalid_geometry(radius, spacing):
    for xp in (np, jnp):
        with pytest.raises(ValueError):
            circular_reference(radius, spacing, xp=xp)


def test_single_point_completes_on_arrival():
    planner = OraclePlannerRL(env(), radius=0.)
    planner.get_reference(jnp.stack([planner.reference_points[0], jnp.ones(3) * 100]))
    np.testing.assert_array_equal(planner.completed_path, [True, False])


def test_resets_preserve_unselected_environments():
    planner = OraclePlannerRL(env())
    planner.reference_point_indices = jnp.array([5, 7])
    planner.completed_path = jnp.array([True, True])
    planner.reset([True, False])
    np.testing.assert_array_equal(planner.reference_point_indices, [0, 7])
    np.testing.assert_array_equal(planner.completed_path, [False, True])
    planner.reset()
    np.testing.assert_array_equal(planner.reference_point_indices, [0, 0])
    np.testing.assert_array_equal(planner.completed_path, [False, False])
    single = OraclePlanner(env())
    single.idx_reference_point = 12
    single.reset()
    assert single.idx_reference_point == 0


@pytest.mark.parametrize("value", [0., -1., np.nan, np.inf])
def test_oracle_rejects_invalid_clearance(value):
    for cls in (OraclePlanner, OraclePlannerRL):
        with pytest.raises(ValueError, match="clearance_dist"):
            cls(env(), clearance_dist=value)


def test_oracle_overlay_is_submitted_once():
    from unittest.mock import Mock
    e = env()
    e.visualization = Mock()
    planner = OraclePlanner(e)
    planner.get_reference(np.ones(3))
    planner.get_reference(np.ones(3))
    assert e.visualization.set_overlay.call_count == 1
