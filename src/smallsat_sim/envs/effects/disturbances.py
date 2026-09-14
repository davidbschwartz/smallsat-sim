"""External wrench effects, their pure JAX state operations and host adapters."""

from .collections import EffectCollection
from smallsat_sim.envs.effects.custom import (
    CustomEffect, CustomEffectState,
)
from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional, Tuple
import jax
import jax.numpy as jnp


class DisturbanceStatus(Enum):
    """
    Document type of active disturbance.
    0 := Constant force disturbance
    """

    CONSTANT_FORCE = 0


@dataclass
class DisturbanceState:
    rng: jnp.ndarray
    active_mask: jnp.ndarray
    start_times: jnp.ndarray
    params: dict


jax.tree_util.register_dataclass(DisturbanceState)


def constant_force_apply_from_state(
    state: Optional[DisturbanceState],
    timestamp: float,
    const_force: jnp.ndarray,
) -> Tuple[jnp.ndarray, Optional[DisturbanceState]]:
    if state is None or const_force is None:
        return jnp.zeros_like(const_force), state

    active_mask = state.active_mask.astype(const_force.dtype)
    time_mask = (timestamp >= state.start_times).astype(const_force.dtype)
    force = const_force * active_mask[:, None] * time_mask[:, None]
    return force, state


def constant_force_activate_state(
    state: Optional[DisturbanceState],
    env_indices: jnp.ndarray,
    start_times: jnp.ndarray,
    const_force: jnp.ndarray,
    rng: jnp.ndarray,
) -> DisturbanceState:
    num_envs = const_force.shape[0]
    if state is None:
        active_mask = jnp.zeros((num_envs,), dtype=bool)
        current_start_times = jnp.zeros((num_envs,), dtype=start_times.dtype)
    else:
        active_mask = state.active_mask
        current_start_times = state.start_times

    if env_indices is not None and env_indices.size > 0:
        active_mask = active_mask.at[env_indices].set(True)
        current_start_times = current_start_times.at[env_indices].set(
            start_times[env_indices]
        )

    return DisturbanceState(
        rng=rng,
        active_mask=active_mask,
        start_times=current_start_times,
        params={"const_force": const_force},
    )


class DisturbanceList(EffectCollection):
    """Sum external wrench contributions in registration order."""

    def __init__(self, disturbances: list) -> None:
        super().__init__()
        self.disturbances = disturbances

        # Initialize look-up dictionary for keycodes and disturbances
        self.keycode_dict = {
            "/": {
                "description": "const_force_disturbance",
                "type": 0,
                "warning": "Could not activate constant force disturbance. "
                "No ConstantForceDisturbance module defined.",
            }
        }

    def __iter__(self):
        return iter(self.disturbances)

    def apply(self, timestamp: Optional[float] = 0.0):
        # Net force vector of all the disturbances in the list
        net_force = jnp.zeros(6)

        for disturbance in self.disturbances:
            if isinstance(disturbance, CustomEffect):
                net_force += disturbance.apply_wrench(timestamp)
            else:
                net_force += disturbance.apply(timestamp)

        return net_force


class ConstantForceDisturbance:
    """
    Applies a constant force disturbance in a specified direction. If an argument is 'None', its value is randomly chosen.
    """

    @property
    def _key(self):
        return self.state.rng

    @property
    def start_times(self):
        return self.state.start_times

    @property
    def const_force(self):
        return self.state.params["const_force"]

    def _split_keys(self, count: int = 1) -> jnp.ndarray:
        """
        Split the internal RNG key and return ``count`` subkeys.
        """
        if count < 1:
            raise ValueError("count must be >= 1")
        splits = jax.random.split(self._key, count + 1)
        self.state = replace(self.state, rng=splits[0])
        return splits[1:]

    def __init__(
        self,
        env_config,
        key: jnp.ndarray,
        magnitude: jnp.ndarray | None = None,
        direction: jnp.ndarray | None = None,
    ) -> None:
        self.state = DisturbanceState(key, jnp.zeros(0, dtype=bool), jnp.zeros(0), {})

        self.failure_type = DisturbanceStatus.CONSTANT_FORCE

        # Find out how many environments there are
        if hasattr(env_config, "environment"):
            self.num_envs = env_config.environment.num_envs
        else:
            self.num_envs = 1

        # Randomize the force magnitude over the enviroments
        if magnitude is None:
            mag_key = self._split_keys(1)[0]
            magnitude = jax.random.uniform(
                mag_key, (self.num_envs, 1), minval=0, maxval=0.2
            )

        # Randomize the force direction over the enviroments
        if direction is None:
            dir_keys = self._split_keys(3)
            x = jax.random.uniform(dir_keys[0], (self.num_envs,), minval=0, maxval=1)
            y = jax.random.uniform(dir_keys[1], (self.num_envs,), minval=0, maxval=1)
            z = jax.random.uniform(dir_keys[2], (self.num_envs,), minval=0, maxval=1)
            direction = jnp.array([x, y, z]).transpose()

        # Normalize all force directions in one vectorized operation. A Python
        # loop here is very expensive when thousands of RL envs are created.
        direction_norm = jnp.linalg.norm(direction, axis=1, keepdims=True)
        direction = direction / jnp.maximum(direction_norm, 1e-8)

        # Save constant disturbance as 6D array
        force, torque = magnitude * direction, jnp.zeros((self.num_envs, 3))
        self.state = DisturbanceState(
            rng=self._key,
            active_mask=jnp.zeros((self.num_envs,), dtype=bool),
            start_times=jnp.zeros(self.num_envs),
            params={
                "const_force": jnp.concatenate((force, torque), axis=1),
            },
        )

    def apply(self, timestamp: Optional[float] = 0.0) -> jnp.ndarray:
        return constant_force_apply_from_state(self.state, timestamp, self.const_force)[0]

    def const_force_disturbance(self, disturbed_envs=None, start_time=0.0):
        """Activate sampled wrenches; omitted targets select all rows in random order."""
        if disturbed_envs is None:
            key = self._split_keys(1)[0]
            disturbed_envs = jax.random.permutation(key, self.num_envs)
        self.activate(jnp.asarray(disturbed_envs, dtype=jnp.int32), start_time=start_time)

    def activate(self, env_indices, *, start_time=0.0, wrenches=None):
        force = self.const_force
        if wrenches is not None:
            force = force.at[env_indices].set(wrenches)
        self.state = constant_force_activate_state(
            self.state, env_indices,
            self.start_times.at[env_indices].set(start_time), force, self._key,
        )

    def deactivate_const_force_disturbance(self) -> None:
        """
        Reset disturbance.
        """
        self.state = replace(self.state, active_mask=jnp.zeros_like(self.state.active_mask))

    def key_callback(self, keycode: Optional[int] = None) -> None:
        # Call correct method for key callbacks
        self.const_force_disturbance()


def __getattr__(name):
    # Keep old checkpoint imports lazy.
    if name in ('disturbance_state_to_serializable', 'disturbance_state_from_serializable'):
        from . import serialization
        return getattr(serialization, name)
    raise AttributeError(name)
