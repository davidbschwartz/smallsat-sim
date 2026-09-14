"""Classical CubeSat MuJoCo environment."""

from smallsat_sim.envs.base_env import BaseEnv


class CubesatEnv(BaseEnv):
    def __init__(self, args, *, config=None, vehicle=None, run_name="default") -> None:
        self.run_name = run_name
        # Load necessary config files
        self.env_cfg, self.model_cfg = self._load_cfg(
            env_name="cubesat", model_name="cubesat",
            vehicle=vehicle, config=config,
        )
        super().__init__(args=args)
