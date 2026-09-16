"""Single-vehicle actuator effects with environment-owned NumPy RNGs.

Time-varying sample faults and the GPyTorch sampler serve classical experiments.
Batched JAX fault application lives in state.py and vectorized.py."""

from .collections import ActuatorEffects
from .catalog import gp_failure_modes
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional
import numpy as np
import jax.numpy as jnp

from smallsat_sim.model.vehicle import VehicleSpec
import gpytorch
import torch
import warnings
from copy import deepcopy

from scipy.interpolate import interp1d


class PerturbationStatus(Enum):
    """
    Document type of active perturbation.
    0 := Thruster fully operational
    1 := Thruster is stuck off
    2 := Thruster is stuck on
    3 is reserved for the removed sampled-trajectory prototype.
    """

    OPERATIONAL = 0
    STUCK_OFF = 1
    STUCK_ON = 2
    FAULTY_VALVE = 4
    SATURATED_THRUST = 5
    THRUST_INSTABILITY = 6


class Perturbation(ABC):
    """
    Base class for all perturbations applied to the model.
    Perturbations are defined as changes to the model's
    dynamics such as a mismatch between desired thrust versus
    actual thrust.
    """

    def __init__(self, model_config: VehicleSpec, *, verbose: bool = False, rng=None) -> None:
        # Extract the number of thrusters
        self.rng = np.random.RandomState() if rng is None else rng
        self.nu = len(model_config.actuators)

        # Thruster mask to document the operational status of thrusters
        self.thruster_mask = jnp.full(self.nu, PerturbationStatus.OPERATIONAL.value)
        self.model_config = model_config
        self.verbose = verbose

    def select_thruster(self, index: Optional[int]) -> int:
        """
        Return index if one is provided, randomly select one of the working thrusters otherwise.
        """
        if isinstance(index, int):
            return index
        else:
            working_thrusters = jnp.where(
                self.thruster_mask == PerturbationStatus.OPERATIONAL.value
            )[0]
            random_index = self.rng.choice(working_thrusters)
            return random_index

    @abstractmethod
    def apply(
        self, input: jnp.ndarray, timestamp: Optional[float] = None
    ) -> jnp.ndarray:
        pass

    @abstractmethod
    def key_callback(self, keycode: Optional[int] = None) -> None:
        pass


class PerturbationList(ActuatorEffects):
    status_type = PerturbationStatus


class StuckOffThrusters(Perturbation):
    """
    This perturbation completly shuts off/fails thrusters.
    """

    def __init__(self, model_config: VehicleSpec, *, verbose: bool = False, rng=None) -> None:
        super().__init__(model_config, verbose=verbose, rng=rng)

        self.failure_type = PerturbationStatus.STUCK_OFF
        self.start_times = jnp.zeros(self.nu)

    def apply(
        self, input: jnp.ndarray, timestamp: Optional[float] = None
    ) -> jnp.ndarray:

        # Do a elementwise AND operation
        stuck_off = jnp.logical_and(
            self.thruster_mask == PerturbationStatus.STUCK_OFF.value,
            timestamp >= self.start_times,
        )

        # Shut off respective thrusters
        input = input.at[stuck_off].set(0.0)

        return input

    def stuck_off_thruster(
        self, index: Optional[int] = None, start_time: Optional[float] = 0.0
    ) -> None:
        """
        Method to shut off a random thruster or a specific one if provided.
        """
        thruster_index = self.select_thruster(index)
        start_time_value = 0.0 if start_time is None else start_time
        self.thruster_mask = self.thruster_mask.at[thruster_index].set(
            PerturbationStatus.STUCK_OFF.value
        )
        self.start_times = self.start_times.at[thruster_index].set(start_time_value)
        if self.verbose:
            print(
                f"Thruster {thruster_index} is stuck off starting at {start_time} seconds."
            )

    def key_callback(self, keycode: Optional[int] = None) -> None:
        # Call correct method for key callbacks
        self.stuck_off_thruster(index=None)


class StuckOnThrusters(Perturbation):
    """
    Thrusters unable to be turned off.
    """

    def __init__(self, model_config: VehicleSpec, *, verbose: bool = False, rng=None) -> None:
        super().__init__(model_config, verbose=verbose, rng=rng)

        self.model_config = model_config

        self.failure_type = PerturbationStatus.STUCK_ON
        self.start_times = jnp.zeros(self.nu)

    def apply(
        self, input: jnp.ndarray, timestamp: Optional[float] = None
    ) -> jnp.ndarray:
        # Find where to apply the perturbation
        stuck_on_thrusters_indices = jnp.where(
            jnp.logical_and(
                self.thruster_mask == PerturbationStatus.STUCK_ON.value,
                timestamp >= self.start_times,
            )
        )[0]

        # Apply the perturbation
        for idx in stuck_on_thrusters_indices:
            thruster_idx = int(idx)
            max_force = self.model_config.actuators[
                thruster_idx
            ].forcerange[1]
            input = input.at[thruster_idx].set(max_force)  # Max. thruster force

        return input

    def stuck_on_thruster(
        self, index: Optional[int] = None, start_time: Optional[float] = 0.0
    ) -> None:
        """
        Method to unable a random thruster or a specific one if provided, to shut off.
        """
        thruster_index = self.select_thruster(index)
        start_time_value = 0.0 if start_time is None else start_time
        self.thruster_mask = self.thruster_mask.at[thruster_index].set(
            PerturbationStatus.STUCK_ON.value
        )
        self.start_times = self.start_times.at[thruster_index].set(start_time_value)
        if self.verbose:
            print(f"Thruster {thruster_index} is stuck on.")

    def key_callback(self, keycode: Optional[int] = None) -> None:
        # Call correct method for key callbacks
        self.stuck_on_thruster(index=None)


class ThrusterFailureSimulator:
    """Sample a bounded command-to-thrust curve for a supported GP fault mode."""

    def __init__(
        self,
        num_points=100,
        upper_bound=0.6,
        subset_size=40,
        valve_min=0.1,
        valve_max=0.5,
        rng=None,
    ):
        self.rng = np.random.RandomState() if rng is None else rng
        self.num_points = num_points
        self.upper_bound = upper_bound
        self.subset_size = subset_size
        self.valve_min = valve_min
        self.valve_max = valve_max
        self.demanded_force = torch.linspace(
            0, upper_bound, num_points
        )  # Values between 0 and upper_bound
        self.failure_modes = self._get_failure_modes()

    class ThrusterFailureGPModel(gpytorch.models.ExactGP):
        def __init__(
            self,
            train_x,
            train_y,
            likelihood,
            kernel_type="RBF",
            lengthscale=0.2,
            outputscale=1.0,
        ):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = self._choose_kernel(kernel_type, lengthscale)
            self.covar_module.outputscale = outputscale

        def _choose_kernel(self, kernel_type, lengthscale):
            if kernel_type == "RBF":
                return gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(lengthscale=lengthscale)
                )
            elif kernel_type == "Matern":
                return gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.MaternKernel(nu=1.5, lengthscale=lengthscale)
                )
            raise ValueError(f"Unsupported kernel type: {kernel_type}")

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def generate_failure_data(self, failure_type):
        actual_force = self._get_actual_force(failure_type)
        subset_indices = self._select_subset_indices()
        demanded_force_subset = self.demanded_force[subset_indices]
        actual_force_subset = actual_force[subset_indices]

        # Initialize GP model and likelihood
        params = self.failure_modes[failure_type]
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = torch.tensor(1e-4)

        model = self.ThrusterFailureGPModel(
            demanded_force_subset,
            actual_force_subset,
            likelihood,
            kernel_type=params["kernel"],
            lengthscale=params["lengthscale"],
            outputscale=params["outputscale"],
        )

        # No training needed as hyperparameters are manually set
        x_test = torch.linspace(0, self.upper_bound, self.num_points)
        sampled_function = self._sample_gp_function(
            model, likelihood, x_test, failure_type
        )

        sampled_function = torch.clamp(sampled_function, 0, self.upper_bound)

        return x_test, sampled_function

    def _normal_like(self, value):
        return torch.as_tensor(
            self.rng.normal(size=tuple(value.shape)), dtype=value.dtype, device=value.device,
        )

    def _get_actual_force(self, failure_type):
        if failure_type == PerturbationStatus.SATURATED_THRUST:
            return torch.where(
                self.demanded_force < 0.2 * self.upper_bound,
                self.demanded_force,
                0.2 * self.upper_bound
                + 0.0005
                * self._normal_like(self.demanded_force)
                * (self.upper_bound - self.demanded_force),
            )
        elif failure_type == PerturbationStatus.FAULTY_VALVE:
            valve_min, valve_max = self.valve_min, self.valve_max
            lower_threshold = 0.67 * self.upper_bound
            upper_threshold = 0.73 * self.upper_bound
            actual_force = torch.full_like(self.demanded_force, valve_min)
            linear_region = (self.demanded_force >= lower_threshold) & (
                self.demanded_force <= upper_threshold
            )
            actual_force[linear_region] = valve_min + (
                (self.demanded_force[linear_region] - lower_threshold)
                / (upper_threshold - lower_threshold)
            ) * (valve_max - valve_min)
            actual_force[self.demanded_force > upper_threshold] = valve_max
            return actual_force
        elif failure_type == PerturbationStatus.THRUST_INSTABILITY:
            return (
                self.demanded_force
                - 0.1 * self.upper_bound * torch.sin(10 * self.demanded_force)
                + 0.05 * self.upper_bound * self._normal_like(self.demanded_force)
            )
        else:
            raise ValueError(f"Unknown failure type: {failure_type}")

    def _select_subset_indices(self):
        indices = self.rng.choice(
            range(1, self.num_points - 1), size=self.subset_size - 2, replace=False
        )
        indices = np.concatenate(([0], indices, [self.num_points - 1]))
        return np.sort(indices)

    def _sample_gp_function(self, gp_model, likelihood, demanded_force, failure_type):
        gp_model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = gp_model(demanded_force)

        if failure_type == PerturbationStatus.SATURATED_THRUST:
            return observed_pred.mean
        else:
            return observed_pred.sample(base_samples=self._normal_like(observed_pred.mean))

    def _get_failure_modes(self):
        return gp_failure_modes(PerturbationStatus)


class GPPerturbation(Perturbation):
    def __init__(self, model_config, failure_type=None, *, verbose: bool = False, rng=None) -> None:
        super().__init__(model_config, verbose=verbose, rng=rng)

        self.failure_type = failure_type if failure_type is not None else self.failure_type
        self.start_times = jnp.zeros(self.nu)
        self.thruster_list = deepcopy(model_config.actuators)

        # Store interpolation functions for each thruster
        self.interpolations = [None] * self.nu

    def apply(
        self, input: jnp.ndarray, timestamp: Optional[float] = None
    ) -> jnp.ndarray:
        # Do a elementwise AND operation
        faulty = jnp.logical_and(
            self.thruster_mask == self.failure_type.value,
            timestamp >= self.start_times,
        )

        # Apply interpolation only to the affected thrusters with active perturbations
        if jnp.any(faulty):
            # Gather input values for the affected thrusters
            affected_inputs = input[faulty]

            # Apply the respective interpolation functions
            interpolated_values = jnp.asarray(
                [
                    self.interpolations[int(thruster_idx)](affected_inputs[idx])
                    for idx, thruster_idx in enumerate(jnp.where(faulty)[0])
                ]
            )

            # Update the input array with interpolated values
            input = input.at[faulty].set(interpolated_values)

        return input

    def register_perturbation(
        self,
        index: Optional[int] = None,
        start_time: Optional[float] = 0.0,
        valve_min: Optional[float] = None,
        valve_max: Optional[float] = None,
    ) -> None:
        """
        Register a perturbation and store the interpolation data.
        """
        if self.failure_type == PerturbationStatus.FAULTY_VALVE and (valve_min is None or valve_max is None):
            warnings.warn("No min or max value set for the FaultyValve perturbation.", UserWarning)


        thruster_index = self.select_thruster(index)
        start_time_value = 0.0 if start_time is None else start_time
        self.thruster_mask = self.thruster_mask.at[thruster_index].set(
            self.failure_type.value
        )
        self.start_times = self.start_times.at[thruster_index].set(start_time_value)

        if valve_min is None:
            valve_min = 0.15 * self.thruster_list[thruster_index].ctrlrange[-1]

        if valve_max is None:
            valve_max = 0.8 * self.thruster_list[thruster_index].ctrlrange[-1]

        # Get the GP-data
        if valve_min is not None and valve_max is not None:
            x_data, y_data = ThrusterFailureSimulator(
                upper_bound=self.thruster_list[thruster_index].ctrlrange[-1],
                valve_min=valve_min,
                valve_max=valve_max,
                rng=self.rng,
            ).generate_failure_data(self.failure_type)

        # Create the interpolation function for the affected thruster
        self.interpolations[thruster_index] = interp1d(
            x_data, y_data, kind="linear", fill_value="extrapolate"
        )

        if self.verbose:
            print(
                f"Thruster {thruster_index} is affected by a faulty valve starting at {start_time} seconds."
            )

    def key_callback(self, keycode: Optional[int] = None) -> None:
        pass


class FaultyValve(GPPerturbation):
    failure_type = PerturbationStatus.FAULTY_VALVE


class SaturatedThrust(GPPerturbation):
    failure_type = PerturbationStatus.SATURATED_THRUST


class ThrustInstability(GPPerturbation):
    failure_type = PerturbationStatus.THRUST_INSTABILITY
