"""Shared circular geometry for the NumPy and JAX oracle planners."""

import numpy as np
from smallsat_sim.planners.validation import positive_finite


def circular_reference(radius, spacing, plane="yz", offsets=(0.0, 0.0, 0.0), *, xp=np):
    """Build waypoints, preserving the starting phase used by each plane."""
    positive_finite("spacing", spacing)
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("radius must be finite and nonnegative")
    if np.shape(offsets) != (3,) or not np.all(np.isfinite(offsets)):
        raise ValueError("offsets must contain three finite coordinates")
    if plane not in {"xy", "yz", "xz"}:
        raise ValueError(f"Unsupported plane '{plane}'. Use 'xy', 'yz', or 'xz'.")
    if radius <= 0:
        return xp.asarray([offsets])
    num_points = int((2 * np.pi * radius) // spacing)
    if num_points < 1:
        raise ValueError("Radius too small for the given spacing.")
    angles = xp.arange(num_points) * spacing / radius
    cosine = radius * xp.cos(angles)
    sine = radius * xp.sin(angles)
    zero = xp.zeros_like(angles)
    coordinates = {
        "xy": (cosine, sine, zero),
        "yz": (zero, sine, cosine),
        "xz": (cosine, zero, sine),
    }[plane]
    return xp.stack(coordinates, axis=1) + xp.asarray(offsets)


def closest_point_on_reference(points, point, spacing, *, xp=np):
    """Project onto adjacent segments and return cumulative polyline arc lengths.

    Outputs retain a batch axis. ``spacing`` is retained for API compatibility;
    progress uses actual segment lengths, including the closing segment.
    """
    points = xp.asarray(points)
    point = xp.atleast_2d(point)
    if point.shape[1] < 3:
        raise ValueError("point must have at least 3 elements per row.")

    pos = point[:, :3]
    num_points = points.shape[0]

    # Compute distances from pos to each reference point
    distances = xp.linalg.norm(
        pos[:, None, :] - points[None, :, :], axis=2
    )  # (N, num_points)
    i_closest = xp.argmin(distances, axis=1)

    # Indices for previous and next points (closed loop)
    i_prev = (i_closest - 1) % num_points
    i_next = (i_closest + 1) % num_points

    # Candidate 1: projection onto segment prev -> closest
    A1 = points[i_prev]
    B1 = points[i_closest]
    v1 = B1 - A1
    dot1 = xp.sum((pos - A1) * v1, axis=1)
    norm_sq1 = xp.sum(v1 * v1, axis=1)
    valid1 = norm_sq1 > 1e-8
    t1 = xp.where(valid1, dot1 / xp.where(valid1, norm_sq1, 1.0), 0.0)
    t1 = xp.clip(t1, 0.0, 1.0)
    proj1 = A1 + t1[:, None] * v1
    candidate1 = xp.where(valid1[:, None], proj1, B1)

    # Candidate 2: projection onto segment closest -> next
    A2 = points[i_closest]
    B2 = points[i_next]
    v2 = B2 - A2
    dot2 = xp.sum((pos - A2) * v2, axis=1)
    norm_sq2 = xp.sum(v2 * v2, axis=1)
    valid2 = norm_sq2 > 1e-8
    t2 = xp.where(valid2, dot2 / xp.where(valid2, norm_sq2, 1.0), 0.0)
    t2 = xp.clip(t2, 0.0, 1.0)
    proj2 = A2 + t2[:, None] * v2
    candidate2 = xp.where(valid2[:, None], proj2, A2)

    # Pick the closer projection for each query
    d1 = xp.linalg.norm(pos - candidate1, axis=1)
    d2 = xp.linalg.norm(pos - candidate2, axis=1)
    choose_first = d1 <= d2
    best_candidate = xp.where(choose_first[:, None], candidate1, candidate2)

    lengths = xp.linalg.norm(xp.roll(points, -1, axis=0) - points, axis=1)
    starts = xp.concatenate([xp.zeros_like(lengths[:1]), xp.cumsum(lengths)[:-1]])
    arc_lengths = xp.where(
        choose_first, starts[i_prev] + t1 * lengths[i_prev],
        starts[i_closest] + t2 * lengths[i_closest],
    )
    total = xp.sum(lengths)
    arc_lengths = arc_lengths % xp.where(total > 0, total, 1.0)

    return best_candidate, arc_lengths
