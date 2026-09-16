"""Prepare configured faults and disturbances for environment construction."""

from .catalog import BUILTIN_FAULTS, FAULTS_BY_NAME
from .collections import ActuatorEffects
from dataclasses import replace
import warnings
import sys
from typing import Optional

import jax
import jax.numpy as jnp

from smallsat_sim.configuration import Settings
from smallsat_sim.envs.effects.gp import ThrusterFailureSimulator, _derive_gp_resolution
from smallsat_sim.envs.effects.actuator_kernels import BatchedFaultState, PerturbationStatus, gp_register_state
from smallsat_sim.envs.effects.catalog import apply_actuator_effect
from smallsat_sim.model.vehicle import VehicleSpec


class ThrusterOccupancy:
    """Read-only view of scheduled actuators, derived from registered effect states."""

    def __init__(self, num_envs: int, num_thrusters: int) -> None:
        self.shape = (num_envs, num_thrusters)
        self.effects = []

    @property
    def mask(self):
        mask = jnp.zeros(self.shape, dtype=jnp.int32)
        for effect in self.effects:
            state = effect.state
            if state is not None and state.failure_value is not None:
                mask = jnp.where(
                    state.thruster_mask == state.failure_value, state.failure_value, mask
                )
        return mask


class BatchedFault:
    """Select actuator targets and prepare sampled state between rollouts."""

    def __init__(
        self,
        env_config: Settings,
        model_config: VehicleSpec,
        key: jnp.ndarray,
        *,
        occupancy: ThrusterOccupancy | None = None,
    ) -> None:
        # Extract the number of thrusters
        self.nu = len(model_config.actuators)

        # Find out how many environments there are
        if hasattr(env_config, "environment"):
            self.num_envs = env_config.environment.num_envs
        else:
            self.num_envs = 1

        self.occupancy = (
            ThrusterOccupancy(self.num_envs, self.nu) if occupancy is None else occupancy
        )
        if self.occupancy.shape != (self.num_envs, self.nu):
            raise ValueError("Thruster occupancy shape does not match environment and vehicle")

        self.verbose = getattr(env_config.sim, "verbose", False)
        self.state: Optional[BatchedFaultState] = BatchedFaultState(
            rng=key,
            thruster_mask=jnp.zeros(self.occupancy.shape, dtype=jnp.int32),
            failure_value=self.failure_type.value,
            start_times=jnp.zeros(self.occupancy.shape),
        )
        self.occupancy.effects.append(self)

    def restore_state(self, state: BatchedFaultState | None) -> None:
        self.state = state

    @property
    def _key(self):
        return self.state.rng

    @property
    def start_times(self):
        return self.state.start_times

    @property
    def stuck_on_force(self):
        return self.state.max_thruster_force

    def _activate_targets(self, envs, thrusters, start_time, **changes):
        """Install targets in this effect; occupancy immediately reflects the new state."""
        mask = self.state.thruster_mask.at[envs, thrusters].set(self.failure_type.value)
        starts = self.state.start_times.at[envs, thrusters].set(
            0.0 if start_time is None else start_time
        )
        self.state = replace(self.state, thruster_mask=mask, start_times=starts, **changes)

    def apply(self, input, timestamp=0.0):
        control = jnp.asarray(input).reshape(self.num_envs, self.nu)
        return apply_actuator_effect(self.state, control, timestamp)[0]

    def _split_keys(self, key: jnp.ndarray | None, count: int) -> jnp.ndarray:
        """
        Split either the provided key or the internal PRNG.
        Returns ``count`` subkeys and updates internal state to the residual key.
        """
        if count < 1:
            raise ValueError("count must be >= 1")
        source = self._key if key is None else key
        splits = jax.random.split(source, count + 1)
        self.state = replace(self.state, rng=splits[0])
        return splits[1:]

    def get_perturbed_envs(
        self,
        key,
        fraction_perturbed_envs: float,
        perturbed_envs: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """
        Return array with envs where a failure occurs. If fraction_perturbed_envs and num_envs are too low, no envs will be perturbed.
        """
        if perturbed_envs is not None:
            return jnp.asarray(perturbed_envs, dtype=jnp.int32)

        fraction = float(
            jax.device_get(jnp.clip(jnp.asarray(fraction_perturbed_envs), 0.0, 1.0))
        )
        num_perturbed_envs = int(fraction * self.num_envs)

        if num_perturbed_envs <= 0:
            return jnp.array([], dtype=jnp.int32)

        operational_thruster_mask = jnp.any(
            self.occupancy.mask == PerturbationStatus.OPERATIONAL.value,
            axis=1,
        )  # Only select envs that still have functioning thrusters
        operational_envs = jnp.where(operational_thruster_mask)[0]

        if operational_envs.size == 0:
            return jnp.array([], dtype=jnp.int32)

        num_perturbed_envs = min(num_perturbed_envs, int(operational_envs.size))
        perm_key = self._split_keys(key, 1)[0]
        shuffled_indices = jax.random.permutation(perm_key, operational_envs)
        return shuffled_indices[:num_perturbed_envs]

    def select_thrusters(
        self,
        key,
        perturbed_envs: jnp.ndarray,
        perturbed_thrusters: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """
        Return thruster indices to fail.
        """

        if perturbed_thrusters is not None:
            return jnp.asarray(perturbed_thrusters, dtype=jnp.int32)

        if perturbed_envs.size == 0:
            return jnp.array([], dtype=jnp.int32)

        def random_operational_thruster(rng_key, row, max_size=self.nu):
            # Get indices of operational thrusters (pad with -1 if not enough)
            operational_thrusters = jnp.where(
                row == PerturbationStatus.OPERATIONAL.value,
                size=max_size,
                fill_value=-1,
            )[0]

            # Replace invalid (-1) indices with a large number so they won't be selected
            valid_mask = operational_thrusters != -1
            masked_indices = jnp.where(valid_mask, operational_thrusters, max_size)
            num_valid = valid_mask.sum()

            def choose_valid(_):
                probs = valid_mask.astype(jnp.float32) / num_valid
                return jax.random.choice(rng_key, masked_indices, p=probs)

            def no_valid(_):
                return jnp.int32(-1)

            return jax.lax.cond(num_valid > 0, choose_valid, no_valid, operand=None)

        selected_envs = self.occupancy.mask[perturbed_envs]
        subkeys = self._split_keys(key, selected_envs.shape[0])

        vec_random_operational_thruster = jax.vmap(
            random_operational_thruster, in_axes=(0, 0)
        )
        selected_thrusters = vec_random_operational_thruster(subkeys, selected_envs)

        return selected_thrusters

    def select_shared_operational_thruster(
        self,
        key: jnp.ndarray,
        perturbed_envs: jnp.ndarray,
        preferred_thruster: int | None = None,
    ) -> int | None:
        """
        Select one thruster index that is operational across all target environments.
        Returns `None` if no shared operational thruster exists.
        """
        if perturbed_envs.size == 0:
            return None

        selected_envs = self.occupancy.mask[perturbed_envs]
        shared_operational = jnp.all(
            selected_envs == PerturbationStatus.OPERATIONAL.value, axis=0
        )
        valid_thrusters = jnp.where(
            shared_operational, size=self.nu, fill_value=-1
        )[0]
        valid_mask = valid_thrusters != -1
        num_valid = int(valid_mask.sum())
        if num_valid <= 0:
            return None

        if preferred_thruster is not None:
            preferred_idx = int(preferred_thruster)
            if 0 <= preferred_idx < self.nu and bool(shared_operational[preferred_idx]):
                return preferred_idx
            return None

        probs = valid_mask.astype(jnp.float32) / float(num_valid)
        chosen = jax.random.choice(key, valid_thrusters, p=probs)
        return int(jax.device_get(chosen))


class BatchedFaultCollection(ActuatorEffects):
    status_type = PerturbationStatus

    def named(self, name):
        status_id = FAULTS_BY_NAME[name].status_id
        return next(effect for effect in self.perturbations
                    if getattr(effect.state, "failure_value", None) == status_id)


class StuckOffThrusters(BatchedFault):
    """
    Set selected actuator commands to zero after onset.
    """

    failure_type = PerturbationStatus.STUCK_OFF

    def activate(
        self,
        key: jnp.ndarray,
        envs: jnp.ndarray | None = None,
        actuators: jnp.ndarray | None = None,
        start_time: float | None = None,
    ) -> None:
        """
        Method to shut off a random thruster or a specific one if provided.
        """
        env_key, thruster_key = jax.random.split(key)
        stuck_off_envs = self.get_perturbed_envs(env_key, 1.0, envs)
        stuck_off_thrusters = self.select_thrusters(
            thruster_key, stuck_off_envs, actuators
        )
        if actuators is None and bool(jnp.any(stuck_off_thrusters < 0)):
            valid = jnp.broadcast_to(stuck_off_thrusters >= 0, stuck_off_envs.shape)
            stuck_off_thrusters = jnp.broadcast_to(stuck_off_thrusters, stuck_off_envs.shape)[valid]
            stuck_off_envs = stuck_off_envs[valid]

        self._activate_targets(stuck_off_envs, stuck_off_thrusters, start_time)

    def stuck_off_thruster(
        self,
        key: jnp.ndarray,
        perturbed_envs: jnp.ndarray | None = None,
        perturbed_thrusters: jnp.ndarray | None = None,
        start_time: float | None = None,
    ) -> None:
        self.activate(key, perturbed_envs, perturbed_thrusters, start_time=start_time)

    def key_callback(self, keycode: int | None = None) -> None:
        # Call correct method for key callbacks
        callback_key = self._split_keys(None, 1)[0]
        self.activate(callback_key)


class StuckOnThrusters(BatchedFault):
    """
    Thrusters unable to be turned off.
    """

    failure_type = PerturbationStatus.STUCK_ON

    def __init__(
        self,
        env_config: Settings,
        model_config: VehicleSpec,
        key: jnp.ndarray,
        *,
        occupancy: ThrusterOccupancy | None = None,
    ) -> None:
        super().__init__(env_config, model_config, key, occupancy=occupancy)

        self.model_config = model_config

        start_times = self.state.start_times

        self.min_thruster_force = jnp.asarray(
            [
                thruster.forcerange[0]
                for thruster in self.model_config.actuators
            ],
            dtype=start_times.dtype,
        )
        self.max_thruster_force = jnp.asarray(
            [
                thruster.forcerange[1]
                for thruster in self.model_config.actuators
            ],
            dtype=start_times.dtype,
        )
        # Per env/thruster stuck-on force value. Active channels are sampled uniformly
        # within each thruster's physical force range when failure is registered.
        stuck_on_force = jnp.tile(
            self.max_thruster_force[None, :], (self.num_envs, 1)
        )

        self.state = replace(self.state, max_thruster_force=stuck_on_force)

    def activate(
        self,
        key: jnp.ndarray,
        envs: jnp.ndarray | None = None,
        actuators: jnp.ndarray | None = None,
        start_time: float | None = None,
    ) -> None:
        """
        Hold selected actuators at a force sampled within their physical range.
        """
        env_key, thruster_key, force_key = jax.random.split(key, 3)
        stuck_on_envs = self.get_perturbed_envs(env_key, 1.0, envs)

        stuck_on_thrusters = self.select_thrusters(
            thruster_key, stuck_on_envs, actuators
        )
        if actuators is None and bool(jnp.any(stuck_on_thrusters < 0)):
            valid = jnp.broadcast_to(stuck_on_thrusters >= 0, stuck_on_envs.shape)
            stuck_on_thrusters = jnp.broadcast_to(stuck_on_thrusters, stuck_on_envs.shape)[valid]
            stuck_on_envs = stuck_on_envs[valid]
        min_force = self.min_thruster_force[stuck_on_thrusters]
        max_force = self.max_thruster_force[stuck_on_thrusters]
        sampled_force = jax.random.uniform(
            force_key,
            shape=(stuck_on_envs.shape[0],),
            minval=min_force,
            maxval=max_force,
        )
        force = self.state.max_thruster_force.at[
            stuck_on_envs, stuck_on_thrusters
        ].set(sampled_force)
        self._activate_targets(
            stuck_on_envs, stuck_on_thrusters, start_time, max_thruster_force=force
        )

    def stuck_on_thruster(
        self,
        key: jnp.ndarray,
        perturbed_envs: jnp.ndarray | None = None,
        perturbed_thrusters: jnp.ndarray | None = None,
        start_time: float | None = None,
    ) -> None:
        self.activate(key, perturbed_envs, perturbed_thrusters, start_time=start_time)

    def key_callback(self, keycode: int | None = None) -> None:
        callback_key = self._split_keys(None, 1)[0]
        self.activate(callback_key)


class BatchedGPFault(BatchedFault):
    """Prepare command-to-thrust curves for selected actuators."""

    def __init__(
        self,
        env_config: Settings,
        model_config: VehicleSpec,
        key: jnp.ndarray,
        failure_type=None,
        *,
        occupancy: ThrusterOccupancy | None = None,
    ) -> None:
        self.failure_type = failure_type if failure_type is not None else self.failure_type
        super().__init__(env_config, model_config, key, occupancy=occupancy)
        self.thruster_list = model_config.actuators
        (
            self._gp_num_points,
            self._gp_subset_size,
        ) = _derive_gp_resolution(self.nu, self.num_envs)
        self._gp_failure_mode_overrides: Optional[dict] = None
        self._gp_sample_cache: dict[tuple, tuple[jnp.ndarray, jnp.ndarray]] = {}

    def _gp_simulator_kwargs(self, **base_kwargs):
        kwargs = dict(base_kwargs)
        kwargs["num_points"] = self._gp_num_points
        kwargs["subset_size"] = self._gp_subset_size
        if self._gp_failure_mode_overrides:
            kwargs["failure_mode_overrides"] = self._gp_failure_mode_overrides
        return kwargs

    def activate(
        self,
        key: jnp.ndarray,
        envs: jnp.ndarray | None = None,
        actuators: int | None = None,
        start_time: float | None = None,
        valve_min: float | None = None,
        valve_max: float | None = None,
    ) -> None:
        """
        Register a perturbation and store the interpolation data.
        NOTE: the same thruster fails in all chosen envs for now.
        """
        if self.failure_type == PerturbationStatus.FAULTY_VALVE and (valve_min is None or valve_max is None):
            warnings.warn("No min or max value set for the FaultyValve perturbation.", UserWarning)

        env_key, thruster_key = jax.random.split(key)
        gp_perturbed_envs = self.get_perturbed_envs(env_key, 1.0, envs)
        if gp_perturbed_envs.size == 0:
            return

        thruster_index = self.select_shared_operational_thruster(
            thruster_key,
            gp_perturbed_envs,
            preferred_thruster=actuators,
        )
        if thruster_index is None:
            if self.verbose:
                print(
                    "Could not register GP perturbation: no shared operational thruster "
                    "for selected environments."
                )
            return

        if valve_min is None:
            valve_min = 0.15 * self.thruster_list[thruster_index].ctrlrange[-1]

        if valve_max is None:
            valve_max = 0.8 * self.thruster_list[thruster_index].ctrlrange[-1]

        gp_data_key = self._split_keys(None, 1)[0]
        simulator_kwargs = self._gp_simulator_kwargs(
            upper_bound=self.thruster_list[thruster_index].ctrlrange[-1],
            valve_min=valve_min,
            valve_max=valve_max,
        )
        cache_key = (
            int(thruster_index),
            float(simulator_kwargs["upper_bound"]),
            float(valve_min),
            float(valve_max),
            int(simulator_kwargs["num_points"]),
            int(simulator_kwargs["subset_size"]),
            int(self.failure_type.value),
            _settings_key(self._gp_failure_mode_overrides),
        )
        cached = self._gp_sample_cache.get(cache_key)
        if cached is None:
            x_data, y_data = ThrusterFailureSimulator(
                **simulator_kwargs
            ).generate_failure_data(gp_data_key, self.failure_type)
            self._gp_sample_cache[cache_key] = (x_data, y_data)
        else:
            x_data, y_data = cached

        if self.verbose:
            start_time_print = float(0.0 if start_time is None else start_time)
            print(
                "Thruster(s) are affected by a faulty valve starting at "
                f"{start_time_print} seconds."
            )

        sampled_state = gp_register_state(
            self.state, thruster_index, jnp.asarray(x_data), jnp.asarray(y_data)
        )
        self._activate_targets(
            gp_perturbed_envs, thruster_index, start_time,
            gp_x_samples=sampled_state.gp_x_samples,
            gp_y_samples=sampled_state.gp_y_samples,
        )

    def register_perturbation(
        self,
        key: jnp.ndarray,
        perturbed_envs: jnp.ndarray | None = None,
        index: int | None = None,
        start_time: float | None = None,
        valve_min: float | None = None,
        valve_max: float | None = None,
    ) -> None:
        self.activate(key, perturbed_envs, index, start_time=start_time, valve_min=valve_min, valve_max=valve_max)


class FaultyValve(BatchedGPFault):
    failure_type = PerturbationStatus.FAULTY_VALVE


class SaturatedThrust(BatchedGPFault):
    failure_type = PerturbationStatus.SATURATED_THRUST


class ThrustInstability(BatchedGPFault):
    failure_type = PerturbationStatus.THRUST_INSTABILITY


def create_actuator_effects(env_config, vehicle, keys):
    occupancy = ThrusterOccupancy(env_config.environment.num_envs, len(vehicle.actuators))
    effects = [getattr(sys.modules[__name__], definition.adapter)(
                   env_config, vehicle, key, occupancy=occupancy)
               for definition, key in zip(BUILTIN_FAULTS, keys, strict=True)]
    return occupancy, BatchedFaultCollection(effects)


def _settings_key(settings):
    if isinstance(settings, dict):
        return tuple(sorted((str(key), _settings_key(value)) for key, value in settings.items()))
    return settings
