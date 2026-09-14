"""Classical oracle planner for circular reference tracking."""

from smallsat_sim.planners.base_planner import BasePlanner
from smallsat_sim.planners.validation import positive_finite
from smallsat_sim.planners.oracle.geometry import (
    circular_reference,
    closest_point_on_reference,
)

import numpy as np


class OraclePlanner(BasePlanner):
    def __init__(
        self,
        env,
        radius=10.5,
        spacing=0.5,
        clearance_dist=0.2,
        plane: str = "yz",
        x_offset: float = 0.0,
        y_offset: float = 0.0,
        z_offset: float = 0.0,
    ) -> None:
        super().__init__(env)
        positive_finite("clearance_dist", clearance_dist)
        self.radius = radius
        self.spacing = spacing
        self.clearance_dist = clearance_dist
        self.plane = plane
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.z_offset = z_offset

        # Initialize current reference point
        self.idx_reference_point = 0

        # generate a circular reference trajectory around the gateway
        self._generate_reference()

        # visualize the circle
        self.visualize(self.reference_points)

    def reset(self):
        """Restart waypoint tracking."""
        self.idx_reference_point = 0

    def get_reference(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Dedicated method which is called externally
        """
        # Check if current state is close enough
        dist = np.linalg.norm(
            obs[0:3]
            - self.reference_points[
                self.idx_reference_point % len(self.reference_points)
            ]
        )

        # If smallsat is closer than the clearance distance, the next reference point is queried
        if dist < self.clearance_dist:
            self.idx_reference_point += 1

        return (
            self.reference_points[
                self.idx_reference_point % len(self.reference_points)
            ].reshape(3, 1),
            np.array([1, 0, 0, 0]).reshape(4, 1),
        )

    def closest_point_on_trajectory(
        self, point: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """
        Finds the closest point on the trajectory to the given point.
        Returns the position and corresponding arc length.
        """
        best_candidate, arc_lengths = closest_point_on_reference(
            self.reference_points, point, self.spacing
        )

        if best_candidate.shape[0] == 1:
            return best_candidate[0], arc_lengths[0]
        return best_candidate, arc_lengths

    def _generate_reference(self):
        """
        Generates a circular reference trajectory in the configured plane.
        """
        self.reference_points = list(
            circular_reference(
                self.radius,
                self.spacing,
                self.plane,
                (self.x_offset, self.y_offset, self.z_offset),
            )
        )
