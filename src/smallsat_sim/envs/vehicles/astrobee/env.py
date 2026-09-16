"""Classical Astrobee MuJoCo environment."""

from smallsat_sim.envs.base_env import BaseEnv
from smallsat_sim.envs.effects.classical import (
    PerturbationList,
    StuckOffThrusters,
    StuckOnThrusters,
    FaultyValve,
    SaturatedThrust,
    ThrustInstability,
)


class AstrobeeEnv(BaseEnv):
    def __init__(self, args, *, config=None, vehicle=None, run_name="default") -> None:
        self.run_name = run_name
        # Load necessary config files
        self.env_cfg, self.model_cfg = self._load_cfg(
            env_name="astrobee", model_name="astrobee",
            vehicle=vehicle, config=config,
        )
        super().__init__(args=args)

        # Instantiate perturbations
        self.reset_perturbations()

    def reset_perturbations(self) -> None:
        """
        Resets the perturbations
        """
        verbose = getattr(self.env_cfg.sim, "verbose", False)
        self.perturbations = PerturbationList(
            [
                StuckOffThrusters(self.model_cfg, verbose=verbose, rng=self.np_rng),  # 0
                StuckOnThrusters(self.model_cfg, verbose=verbose, rng=self.np_rng),  # 1
                FaultyValve(self.model_cfg, verbose=verbose, rng=self.np_rng),  # 2
                SaturatedThrust(self.model_cfg, verbose=verbose, rng=self.np_rng),  # 3
                ThrustInstability(self.model_cfg, verbose=verbose, rng=self.np_rng),  # 4
            ]
        )
