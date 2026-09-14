"""Vectorized Astrobee environment for reinforcement learning."""

from smallsat_sim.envs.effects.custom import append_custom_effects, validate_custom_effects
from copy import deepcopy
from smallsat_sim.envs.effects.catalog import (
    BUILTIN_FAULTS,
    FaultSpec,
    configured_faults,
    fault_weights,
)
from smallsat_sim.envs.effects.scheduling import (
    schedule_fault,
)
from smallsat_sim.api.registry import get_reward, get_termination

from smallsat_sim.model.vehicle import load_vehicle, validate_vehicle
from smallsat_sim.envs.vec_env import VecEnv
from smallsat_sim.envs.effects.preparation import create_actuator_effects
from smallsat_sim.envs.effects.disturbances import DisturbanceList, ConstantForceDisturbance


class AstrobeeEnvVectorized(VecEnv):
    def __init__(self, args, *, config, vehicle=None, run_name="default") -> None:
        self.run_name = run_name
        validate_custom_effects(config.environment.custom_faults)
        validate_custom_effects(config.environment.custom_disturbances)
        get_termination(config.environment.termination)
        get_reward(config.environment.reward)
        fault_weights(config.environment.failure_distribution)
        self.env_cfg = deepcopy(config)
        self.env_name = "astrobee_rl"
        self.model_name = self.env_cfg.model
        self.model_cfg = (
            load_vehicle(f"vehicles/{self.env_cfg.model}.yaml")
            if vehicle is None else validate_vehicle(vehicle)
        )
        if len(self.model_cfg.bodies) != 1 or config.Bodies.num_bodies != 1:
            raise ValueError("The vectorized free-flyer task requires exactly one free body")
        if config.environment.rollout_backend not in ("freeflyer", "mjx"):
            raise ValueError("rollout_backend must be mjx or freeflyer")
        self.fault_specs = configured_faults(
            config.environment.faults, self.model_cfg, config.environment.num_envs,
        )
        super().__init__(args=args)

        # Instantiate perturbations
        self.thruster_occupancy, self.perturbations = create_actuator_effects(
            self.env_cfg, self.model_cfg, self.next_rng_keys(len(BUILTIN_FAULTS))
        )

        # Instantiate disturbances
        disturbance_key = self.next_rng_keys(1)[0]
        self.disturbances = DisturbanceList(
            [ConstantForceDisturbance(self.env_cfg, disturbance_key)]
        )
        append_custom_effects(self, disturbance=False)
        append_custom_effects(self, disturbance=True)
        self._refresh_effect_states()
        self._schedule_configured_faults()

    def schedule_fault(self, spec: FaultSpec, *, key):
        schedule_fault(self, spec, key=key)

    def _schedule_configured_faults(self):
        for spec in self.fault_specs:
            self.schedule_fault(spec, key=self.next_rng_keys(1)[0])

    def reset_perturbations(self) -> None:
        """
        Resets the perturbations.
        """
        self.thruster_occupancy, self.perturbations = create_actuator_effects(
            self.env_cfg, self.model_cfg, self.next_rng_keys(len(BUILTIN_FAULTS))
        )

        append_custom_effects(self, disturbance=False)
        self._refresh_effect_states()
        self._schedule_configured_faults()

    def reset_disturbances(self) -> None:
        """
        Reset external disturbances before sampling the next rollout.
        """
        disturbance_key = self.next_rng_keys(1)[0]
        self.disturbances = DisturbanceList(
            [ConstantForceDisturbance(self.env_cfg, disturbance_key)]
        )
        append_custom_effects(self, disturbance=True)
        self._refresh_effect_states()
