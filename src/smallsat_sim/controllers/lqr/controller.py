from smallsat_sim.controllers.base_controller import BaseController
from smallsat_sim.envs.base_env import BaseEnv
from smallsat_sim.utils.helpers import (
    calc_lateral_tracking_error,
    calc_attitude_error,
    quat_multiply,
    quat_conjugate,
    Rquat,
)

import numpy as np
import casadi as ca
import control

from scipy.linalg import solve_discrete_are, expm
from scipy.optimize import lsq_linear
from smallsat_sim.controllers.state import body_state


class LQRController(BaseController):
    """
    This is an implementation of a LQR for tracking problems. The reference is provided by the planner.
    """

    def __init__(self, env, planner) -> None:
        # Fetch correct controller config
        self.ctrl_cfg = env.env_cfg.control.LQR

        # Initialize base class
        super().__init__(env, planner, self.ctrl_cfg)

        # Fetch symbolic model
        self.model = env.symbolic_model
        self.f = self.model.f_expl_expr
        self.x = self.model.x
        self.u = self.model.u
        self.nx = self.x.shape[0]
        self.nu = self.u.shape[0]

        self._linearization = ca.Function("lqr_linearization", [self.x, self.u],
                                           [ca.jacobian(self.f, self.x), ca.jacobian(self.f, self.u)])
        self.v_ref = np.zeros((3, 1))
        self.omega_ref = np.zeros((3, 1))
        self.Q = np.asarray(self.ctrl_cfg.cost.Q) + 1e-5 * np.eye(self.nx)
        self.R = np.asarray(self.ctrl_cfg.cost.R)
        self.dt = env.env_cfg.sim.dt * self.ctrl_cfg.control_decimation
        self._gain_key = None
        self._gain = None
        ranges = np.asarray([a.forcerange for a in env.model_cfg.actuators])
        self._lower, self._upper = ranges.T

    def get_lqr_gain(self, x):
        """Linearize in a quaternion tangent chart at the reference, then use ZOH.

        Coordinates are world position, reference-body quaternion vector error,
        body linear velocity and body angular velocity. No singular quaternion
        scalar deletion, even at 180 degree target attitudes.
        """
        key = np.asarray(x, dtype=float).tobytes()
        if key == self._gain_key:
            return self._gain
        q = self._normalize_quat(x[3:7])
        w, v = q[0], q[1:]
        skew = np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])
        lift = np.zeros((13, 12))
        lift[:3, :3] = np.eye(3)
        lift[3:7, 3:6] = np.vstack((-v, w * np.eye(3) + skew))
        lift[7:, 6:] = np.eye(6)
        A, B = (np.asarray(value) for value in self._linearization(x, np.zeros(self.nu)))
        A, B = lift.T @ A @ lift, lift.T @ B
        block = np.zeros((12 + self.nu, 12 + self.nu))
        block[:12, :12], block[:12, 12:] = A, B
        discrete = expm(block * self.dt)
        A, B = discrete[:12, :12], discrete[:12, 12:]
        P = solve_discrete_are(A, B, lift.T @ self.Q @ lift, self.R)
        gain = np.linalg.solve(B.T @ P @ B + self.R, B.T @ P @ A)
        if not np.all(np.isfinite(gain)):
            raise RuntimeError("LQR produced a nonfinite gain")
        self._gain_key, self._gain = key, gain
        return gain

    def _normalize_quat(self, q: np.ndarray) -> np.ndarray:
        eps = 1e-12
        n = np.linalg.norm(q)
        if n < eps:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        return q / n

    def _quat_error(self, q_ref: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Return error quaternion q_err = conj(q_ref) ⊗ q,
        adjusted to the shortest path.
        """
        q_ref = self._normalize_quat(q_ref)
        q = self._normalize_quat(q)
        if np.dot(q_ref, q) < 0:
            q = -q
        q_err = quat_multiply(quat_conjugate(q_ref), q)
        if q_err[0] < 0:
            q_err = -q_err
        return q_err

    def _body_to_inertial_vel(self, q: np.ndarray, v_body: np.ndarray) -> np.ndarray:
        """
        Convert body-frame velocity to inertial frame using quaternion.
        """
        R_ib = Rquat(q)  # body -> inertial
        return (R_ib @ v_body.reshape(3, 1)).reshape(3,)

    def get_control_input(self, env: BaseEnv) -> np.ndarray:
        """
        Calculate the control input based on current observation.
        """

        # Get current state
        x = body_state(env)
        r = x[0:3]
        q = x[3:7]
        v_body = x[7:10]
        omega = x[10:13]

        # Get the reference position
        r_ref, quat_ref = self.planner.get_reference(env.get_obs())
        r_ref = np.asarray(r_ref).reshape(3,)
        quat_ref = self._normalize_quat(np.asarray(quat_ref).reshape(4,))
        ref_body_velocity = np.asarray(Rquat(quat_ref)).T @ self.v_ref.reshape(3,)
        x_ref = np.concatenate(
            (
                r_ref,
                quat_ref,
                ref_body_velocity,
                self.omega_ref.reshape(3,),
            )
        )

        # Compute the LQR gain around the reference state
        self.K = self.get_lqr_gain(x_ref)

        # Build error state (zero at reference)
        q_err = self._quat_error(quat_ref, q)
        eps_err = q_err[1:4]
        x_err = np.concatenate(
            (
                r - r_ref,
                eps_err,
                v_body - ref_body_velocity,
                omega - self.omega_ref.reshape(3,),
            )
        )

        # Compute optimal control signal
        u_opt = -self.K @ x_err

        # LQR optimizes signed inputs; physical thrusters are unilateral.
        matrix = self.model.mixer
        scale = np.maximum(np.linalg.norm(matrix, axis=1), 1e-12)
        weighted = matrix / scale[:, None]
        fixed = self._upper <= self._lower
        command = self._lower.copy()
        if np.any(~fixed):
            target = weighted @ (u_opt - command)
            result = lsq_linear(weighted[:, ~fixed], target,
                                bounds=(np.zeros(np.sum(~fixed)),
                                        (self._upper - self._lower)[~fixed]), tol=1e-10)
            if not result.success or not np.all(np.isfinite(result.x)):
                raise RuntimeError("LQR thrust allocation failed")
            command[~fixed] += result.x
        return command

    def _log(self, run_id: int, timestamp: float, env: BaseEnv) -> None:
        """
        Logs desired quantities if flag is enabled
        """
        if self.has_logger:
            obs_gt = env.get_obs()

            tracking_error = calc_lateral_tracking_error(
                obs=obs_gt, planner=self.planner
            )
            q_ref = np.array([1.0, 0.0, 0.0, 0.0])
            if hasattr(self.planner, "trajectory") and hasattr(
                self.planner.trajectory, "get_intermediate_reference"
            ):
                _, curr_arc_length = self.planner.closest_point_on_trajectory(
                    point=obs_gt[:3]
                )
                q_ref = self.planner.trajectory.get_intermediate_reference(
                    curr_arc_length
                ).attitude
            attitude_error = calc_attitude_error(
                q_ref=np.asarray(q_ref).squeeze(), q=obs_gt[3:7]
            )

            self.logger.log(
                run_id=run_id,
                timestamp=timestamp,
                tracking_error=tracking_error,
                attitude_error=attitude_error,
            )
