"""Classical controller evaluation in the same lightweight scene as RL."""

from smallsat_sim.envs.vehicles.astrobee.env import AstrobeeEnv


class AstrobeeBenchmarkEnv(AstrobeeEnv):
    """Astrobee dynamics and faults with the training scene (no station geometry)."""

    scene = "training"

    def reset_disturbances(self) -> None:
        self.disturbances = None
