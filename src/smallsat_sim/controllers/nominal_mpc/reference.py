"""Quaternion reference representation for the nominal MPC's quadratic cost."""

import numpy as np


class MPCSolverError(RuntimeError):
    """A failed solve has no valid control command to apply."""

    def __init__(self, status):
        self.status = int(status)
        super().__init__(f"MPC solver failed with status {status}; command was rejected")


def require_valid_control(status, command, lower=None, upper=None):
    if status != 0 or not np.all(np.isfinite(command)):
        raise MPCSolverError(status)
    if lower is not None and (np.any(command < np.asarray(lower) - 1e-6)
                              or np.any(command > np.asarray(upper) + 1e-6)):
        raise ValueError("MPC command violates actuator bounds; command was rejected")
    return command


def align_quaternion_reference(current, reference):
    reference = np.asarray(reference)
    return reference * (-1.0 if np.dot(np.ravel(current), reference.ravel()) < 0 else 1.0)


def attitude_cost(current, desired, weights):
    """Sign-invariant body-frame quaternion-vector cost (CasADi expression)."""
    import casadi as ca
    vector = (current[0] * desired[1:4] - desired[0] * current[1:4]
              - ca.cross(current[1:4], desired[1:4]))
    return ca.mtimes([vector.T, ca.DM(np.asarray(weights)[1:4, 1:4]), vector])


def unwrap_progress(wrapped, previous, length):
    """Lift a periodic projection onto the nearest nonnegative lap branch."""
    return max(0., float(wrapped + length * np.floor((previous - wrapped) / length + .5)))


def rk4_step(dynamics, state, control, dt):
    """Classical four-stage RK step, also usable with symbolic state/control."""
    k1 = dynamics(state, control)
    k2 = dynamics(state + dt * k1 / 2, control)
    k3 = dynamics(state + dt * k2 / 2, control)
    k4 = dynamics(state + dt * k3, control)
    return state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
