"""State and actuator contracts shared by classical controllers."""

import numpy as np
from smallsat_sim.utils.helpers import Rquat


def body_state(env):
    """Copy the 13D physical observation, expressing linear velocity in body axes."""
    state = np.asarray(env.get_obs(), dtype=float).reshape(-1)[:13].copy()
    if state.shape != (13,) or not np.all(np.isfinite(state)):
        raise ValueError("Classical controllers require a finite 13D physical state")
    norm = np.linalg.norm(state[3:7])
    if norm < 1e-12:
        raise ValueError("State quaternion must be nonzero")
    state[3:7] /= norm
    frame = env.env_cfg.sim.obs.v_frame
    if frame == "inertial":
        state[7:10] = np.asarray(Rquat(state[3:7])).T @ state[7:10]
    elif frame != "body":
        raise ValueError(f"Unsupported velocity frame: {frame}")
    return state


def actuator_bounds(model):
    """Use declared ranges even when the simulator delegates limiting to effects.

    MuJoCo's default [0,0] with limiting disabled denotes an unspecified range.
    Such site thrusters retain a nonnegative, unbounded command domain.
    """
    lower = np.zeros(model.nu)
    upper = np.full(model.nu, np.inf)
    for ranges, limited in ((model.actuator_forcerange, model.actuator_forcelimited),
                            (model.actuator_ctrlrange, model.actuator_ctrllimited)):
        selected = (ranges[:, 1] > ranges[:, 0]) | np.asarray(limited, dtype=bool)
        lower[selected] = np.maximum(lower[selected], ranges[selected, 0])
        upper[selected] = np.minimum(upper[selected], ranges[selected, 1])
    if np.any(upper < lower):
        raise ValueError("Actuator force/control ranges do not intersect")
    return lower, upper
