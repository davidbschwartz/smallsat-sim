"""Shared validation for planner configuration."""
import numpy as np


def positive_finite(name, value):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def require_contouring_planner(planner):
    """Fail before native solver allocation for unsupported path interfaces."""
    trajectory = getattr(planner, "trajectory", None)
    methods = ("get_intermediate_reference", "_get_start_point_segment",
               "_get_tangent_segment", "_get_start_arc_length_segment")
    if trajectory is None or any(not callable(getattr(trajectory, name, None)) for name in methods):
        raise ValueError("MPCC/GP-MPC require a piecewise-line MissionPlanner trajectory; "
                         "use nominal MPC, LQR or PD for this planner")
    # Their optimization represents a straight line at each stage, not an arc.
    from smallsat_sim.planners.mission.mission import Line
    if any(not isinstance(segment, Line) for segment in trajectory.reference):
        raise ValueError("MPCC/GP-MPC currently support only piecewise-line trajectories")
