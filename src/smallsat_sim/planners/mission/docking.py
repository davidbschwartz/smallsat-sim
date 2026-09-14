"""Docking trajectory planner with bounded approach geometry."""

from __future__ import annotations

import numpy as np

from smallsat_sim.planners.base_planner import BasePlanner
from smallsat_sim.planners.validation import positive_finite


class DockingPlanner(BasePlanner):
    """
    Two-stage docking planner:
    1) Fly to pre-dock point
    2) Fly to final dock point
    """

    def __init__(
        self,
        env,
        pre_dock_position: np.ndarray,
        dock_position: np.ndarray,
        dock_attitude: np.ndarray | None = None,
        switch_distance: float = 0.35,
        dock_distance: float = 0.15,
        approach_speed: float | None = None,
        max_attitude_error: float = 0.25,
        max_speed: float = 0.05,
        max_angular_speed: float = 0.05,
        require_contact: bool = True,
        max_contact_force: float = 8.0,
        contact_body: str = "gateway_full",
        validate_geometry: bool = True,
    ) -> None:
        super().__init__(env)
        positive_finite("switch_distance", switch_distance)
        positive_finite("dock_distance", dock_distance)
        if approach_speed is not None:
            positive_finite("approach_speed", approach_speed)

        self.pre_dock_position = np.asarray(pre_dock_position, dtype=float).reshape(3)
        self.dock_position = np.asarray(dock_position, dtype=float).reshape(3)

        if dock_attitude is None:
            dock_attitude = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.dock_attitude = np.asarray(dock_attitude, dtype=float).reshape(4)

        if not np.all(np.isfinite(np.r_[self.pre_dock_position, self.dock_position, self.dock_attitude])):
            raise ValueError("Docking pose must be finite")
        norm = np.linalg.norm(self.dock_attitude)
        if norm < 1e-12:
            raise ValueError("Docking quaternion must be nonzero")
        self.dock_attitude /= norm
        for name, value in (("max_attitude_error", max_attitude_error), ("max_speed", max_speed),
                            ("max_angular_speed", max_angular_speed), ("max_contact_force", max_contact_force)):
            positive_finite(name, value)
        self.max_attitude_error = max_attitude_error
        self.max_speed = max_speed
        self.max_angular_speed = max_angular_speed
        self.require_contact = require_contact
        self.max_contact_force = max_contact_force
        self.contact_body = contact_body
        self.validate_geometry = validate_geometry
        self.switch_distance = float(switch_distance)
        self.dock_distance = float(dock_distance)
        self.stage = 0
        self.approach_speed = approach_speed
        self.env = env
        self._approach_reference = self.pre_dock_position.copy()
        self._reference_time = None
        self._checked_start = False
        if validate_geometry and hasattr(env, "model"):
            self.check_approach_geometry()
        self.visualize([self.pre_dock_position, self.dock_position], size=(.03, 0, 0))


    def reset(self):
        """Restart at the pre-dock approach stage."""
        self.stage = 0
        self._approach_reference = self.pre_dock_position.copy()
        self._reference_time = None
        self._checked_start = False

    def get_reference(
        self, obs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        position = np.asarray(obs[:3], dtype=float).reshape(3)
        if self.validate_geometry and not self._checked_start and hasattr(self.env, "model"):
            from smallsat_sim.planners.clearance import ClearanceScene
            scene = ClearanceScene(self.env)
            if not scene.segment_is_clear(position, self.pre_dock_position, clearance=0.,
                                          end_quaternion=self.dock_attitude):
                raise ValueError("Initial docking pose or route to pre-dock intersects collision geometry")
            self._checked_start = True

        if self.stage == 0:
            dist_to_pre_dock = np.linalg.norm(position - self.pre_dock_position)
            if dist_to_pre_dock <= self.switch_distance and self._pose_ready(obs):
                self.stage = 1

        target_position = (
            self.pre_dock_position if self.stage == 0 else self.dock_position
        )
        if self.approach_speed is not None:
            now = float(self.env.data.time)
            dt = 0.0 if self._reference_time is None else max(0.0, now - self._reference_time)
            self._reference_time = now
            if self.stage == 1:
                delta = self.dock_position - self._approach_reference
                distance = np.linalg.norm(delta)
                if distance > 0:
                    self._approach_reference += delta * min(1.0, self.approach_speed * dt / distance)
                target_position = self._approach_reference

        return target_position.reshape(3, 1), self.dock_attitude.reshape(4, 1)

    def closest_point_on_trajectory(
        self, point: np.ndarray
    ) -> tuple[np.ndarray, float]:
        point = np.asarray(point, dtype=float).reshape(3)
        start = self.pre_dock_position
        end = self.dock_position

        segment = end - start
        segment_length = np.linalg.norm(segment)
        if segment_length < 1e-12:
            return start.copy(), 0.0

        t = np.dot(point - start, segment) / np.dot(segment, segment)
        t_clipped = float(np.clip(t, 0.0, 1.0))
        closest = start + t_clipped * segment
        arc_length = t_clipped * segment_length

        return closest, arc_length

    def _pose_ready(self, obs):
        obs = np.asarray(obs).ravel()
        if len(obs) < 13 or not np.all(np.isfinite(obs[:13])):
            return False
        norm = np.linalg.norm(obs[3:7])
        if norm < 1e-12:
            return False
        angle = 2 * np.arccos(np.clip(abs(np.dot(obs[3:7] / norm, self.dock_attitude)), 0., 1.))
        return (angle <= self.max_attitude_error
                and np.linalg.norm(obs[7:10]) <= self.max_speed
                and np.linalg.norm(obs[10:13]) <= self.max_angular_speed)

    def check_approach_geometry(self):
        """Reject a colliding pre-dock pose or obstructed exterior approach.

        The final dock-distance region permits intended compliant contact; the
        contact/pose gate below determines capture. No hardware latch is modeled.
        """
        from smallsat_sim.planners.clearance import ClearanceScene
        scene = ClearanceScene(self.env)
        delta = self.pre_dock_position - self.dock_position
        length = np.linalg.norm(delta)
        exterior = self.dock_position + delta * min(1., self.dock_distance / max(length, 1e-12))
        if not scene.segment_is_clear(self.pre_dock_position, exterior,
                                      clearance=0., quaternion=self.dock_attitude):
            raise ValueError("Docking approach intersects spacecraft collision geometry; choose an exterior port")

    def is_docked(self, obs: np.ndarray) -> bool:
        if (not self._pose_ready(obs)
                or np.linalg.norm(np.asarray(obs[:3]) - self.dock_position) > self.dock_distance):
            return False
        if not self.require_contact:
            return True
        if not hasattr(self.env, "model") or not hasattr(self.env, "data"):
            return False
        import mujoco
        from smallsat_sim.planners.clearance import ClearanceScene
        scene = ClearanceScene(self.env)
        try:
            target = int(self.env.model.body(self.contact_body).id)
        except KeyError:
            return False
        target_bodies = {target}
        for body in range(target + 1, self.env.model.nbody):
            if int(self.env.model.body_parentid[body]) in target_bodies:
                target_bodies.add(body)
        active, force = False, 0.
        for contact_index in range(self.env.data.ncon):
            contact = self.env.data.contact[contact_index]
            a, b = int(contact.geom1), int(contact.geom2)
            if a not in scene.vehicle_geoms:
                a, b = b, a
            if a in scene.vehicle_geoms and int(self.env.model.geom_bodyid[b]) in target_bodies:
                wrench = np.zeros(6)
                mujoco.mj_contactForce(self.env.model, self.env.data, contact_index, wrench)
                force += np.linalg.norm(wrench[:3])
                active = active or contact.dist <= 0
        return active and np.isfinite(force) and force <= self.max_contact_force
