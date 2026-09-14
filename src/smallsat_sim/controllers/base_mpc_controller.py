"""Shared MPC controller interface and trajectory buffers."""

from smallsat_sim.envs.base_env import BaseEnv
from smallsat_sim.planners.base_planner import BasePlanner
from smallsat_sim.controllers.base_controller import BaseController

from abc import abstractmethod

import numpy as np


class BaseMPCController(BaseController):
    def __init__(self, env: BaseEnv, planner: BasePlanner, ctrl_cfg: object) -> None:

        if not np.isclose(ctrl_cfg.Ts, env.env_cfg.sim.dt * ctrl_cfg.control_decimation):
            raise ValueError("MPC Ts must equal sim.dt * control_decimation")
        for name in ("Q", "R", "T", "Q_c", "Q_q", "Q_omega"):
            if hasattr(ctrl_cfg.cost, name):
                setattr(ctrl_cfg.cost, name, np.asarray(getattr(ctrl_cfg.cost, name), dtype=float))
        if ctrl_cfg.cost.R.shape != (env.symbolic_model.nu, env.symbolic_model.nu):
            raise ValueError("MPC R must match the spacecraft actuator count")
        super().__init__(env, planner, ctrl_cfg)

        # Save symbolic model here too
        self.symbolic_model = env.symbolic_model
        
        # Control Input callback time
        self.ctrl_input_callback_time = 0
        self.solve_time = 0

    @abstractmethod
    def get_control_input(self, env: BaseEnv) -> np.ndarray:
        """
        Returns the control input
        """
        pass

    @abstractmethod
    def _log(self, run_id: int, timestamp: float, env: BaseEnv) -> None:
        """
        Logs desired quantities if flag is enabled
        """
        pass

    @abstractmethod
    def _visualize_prediction(self) -> None:
        """
        Plot predicted trajectory of MPC in MuJoCo viewer.
        """
        pass

    def _visualize_prediction_renderer(self) -> None:
        """
        Plot predicted trajectory of MPC in MuJoCo renderer.
        """
        pass
