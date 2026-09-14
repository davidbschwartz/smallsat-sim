"""Supply controller observations without allocating a simulator."""

import numpy as np


class StandaloneEnv:
    def __init__(self, env_cfg, model, model_cfg):
        self.env_cfg = env_cfg
        self.env_cfg.sim.obs.v_frame = "inertial"
        self.symbolic_model = model
        self.model_cfg = model_cfg
        self.viewer = None
        self.renderer = None
        self.using_rl = False
        self._obs = None

    def set_obs(self, qpos, velocity, angular_velocity):
        """Set position, quaternion, inertial velocity and body angular velocity."""
        self._obs = np.concatenate((qpos, velocity, angular_velocity))

    @property
    def obs(self):
        return self.get_obs()

    def get_obs(self):
        """Return a copy of the externally supplied observation."""
        if self._obs is None:
            raise RuntimeError("Call set_obs() before requesting an observation")
        return self._obs.copy()
