"""Base planner interface for classical and vectorized environments."""

from typing import Any, Generic, TypeVar
from numpy.typing import NDArray
import numpy as np
import jax.numpy as jnp
from abc import abstractmethod, ABC

from smallsat_sim.envs.base_env import BaseEnv


T = TypeVar("T", np.ndarray, jnp.ndarray)
T1 = TypeVar("T1", tuple[NDArray[Any], NDArray[Any]], jnp.ndarray)
T2 = TypeVar("T2", tuple[NDArray[Any], float], jnp.ndarray)


class BasePlanner(ABC, Generic[T, T1, T2]):
    """
    Base class of planner objects.
    """

    def __init__(self, env: BaseEnv) -> None:
        self.using_rl = env.using_rl
        self.visualization = getattr(env, "visualization", None)

    @abstractmethod
    def get_reference(self, obs: T) -> T1:
        """
        Returns a reference for the position and attitude based on current observations
        """
        pass

    @abstractmethod
    def closest_point_on_trajectory(self, point: T) -> T2:
        """
        Finds the closest point on the trajectory to the given point.
        Returns the position and corresponding arc length.
        """
        pass

    def visualize(self, points, color=(1, 0, 0, 1), size=(.005, 0, 0)):
        """Submit reference geometry to the environment's rendering component."""
        if self.visualization is not None:
            self.visualization.set_overlay("reference", points, color=color, radius=size[0])

    def _visualize_renderer(self, points, color=(1, 0, 0, 1), size=(.005, 0, 0)):
        self.visualize(points, color=color, size=size)
