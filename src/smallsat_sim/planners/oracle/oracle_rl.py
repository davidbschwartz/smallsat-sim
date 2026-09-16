"""Vectorized oracle planner for RL reference tracking."""

from smallsat_sim.planners.base_planner import BasePlanner
from smallsat_sim.planners.validation import positive_finite
from smallsat_sim.planners.oracle.geometry import (
    circular_reference,
    closest_point_on_reference,
)
from smallsat_sim.envs.vec_env import VecEnv

import jax.numpy as jnp


class OraclePlannerRL(BasePlanner):
    def __init__(
        self, env: VecEnv, radius=3.0, spacing=1.0, clearance_dist=0.2
    ) -> None:
        super().__init__(env)
        positive_finite("clearance_dist", clearance_dist)
        self.radius = radius
        self.spacing = spacing
        self.clearance_dist = jnp.full(env.num_envs, clearance_dist)

        # Initialize current reference point
        self.reference_point_indices = jnp.zeros(env.num_envs, dtype=jnp.int32)

        # Flags to indicate whether the agents have completed the path
        self.completed_path = jnp.zeros(env.num_envs, dtype=jnp.bool_)

        # Generate a circular reference trajectory around the gateway
        self._generate_reference()

        # Visualize the circle
        self.visualize(self.reference_point_list)

    def reset(self, mask=None):
        """Restart all environments, or those selected by a boolean mask."""
        if mask is None:
            mask = jnp.ones_like(self.completed_path)
        mask = jnp.asarray(mask, dtype=bool)
        if mask.shape != self.completed_path.shape:
            raise ValueError("reset mask must have shape (num_envs,)")
        self.reference_point_indices = jnp.where(mask, 0, self.reference_point_indices)
        self.completed_path = jnp.where(mask, False, self.completed_path)

    def get_reference(self, obs: jnp.ndarray) -> jnp.ndarray:
        """
        Get the next position and identity attitude for each environment.
        """
        num_points = self.reference_points.shape[0]
        # Get the current reference point for each environment
        ref_points = self.reference_points[self.reference_point_indices % num_points]
        # Compute the distance for each environment
        dist = jnp.linalg.norm(obs[:, 0:3] - ref_points, axis=1)

        # If smallsat is closer than the clearance distance, advance to the next reference point
        self.reference_point_indices = self.reference_point_indices + jnp.where(
            dist < self.clearance_dist, 1, 0
        )

        # Check if the agents have completed the path
        self.completed_path = jnp.where(
            jnp.logical_or(
                self.reference_point_indices >= (1 if num_points == 1 else num_points + 1),
                self.completed_path
            ),
            True,
            False,
        )

        ref_pos = self.reference_points[self.reference_point_indices % num_points]
        ref_quat = jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (obs.shape[0], 1))

        return jnp.concatenate([ref_pos, ref_quat], axis=1)

    def closest_point_on_trajectory(self, point: jnp.ndarray) -> jnp.ndarray:
        """
        Finds the orthogonal projection of the smallsat position onto the trajectory.
        """
        positions, _ = closest_point_on_reference(
            self.reference_points, point, self.spacing, xp=jnp
        )
        return positions

    def _generate_reference(self):
        """Generate an xy circle centered at the gateway's height."""
        self.reference_points = circular_reference(
            self.radius, self.spacing, "xy", (0.0, 0.0, 10.17), xp=jnp
        )
        self.reference_point_list = list(self.reference_points)
