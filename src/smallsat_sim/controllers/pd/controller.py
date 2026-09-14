# (License for _apply_ctrl_constraint function)
# Copyright (c) 2017, United States Government, as represented by the
# Administrator of the National Aeronautics and Space Administration.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

import numpy as np

from smallsat_sim.controllers.base_controller import BaseController
from smallsat_sim.controllers.state import actuator_bounds
from smallsat_sim.controllers.pd.common import (
    calc_B_matrix, compute_control, constrain_thrust, thruster_groups,
)


class PDController(BaseController):
    """Single-environment NumPy adapter for the shared PD calculation."""

    def __init__(self, env, planner):
        ctrl_cfg = env.env_cfg.control.PD
        super().__init__(env, planner, ctrl_cfg)
        self.v_ref = np.zeros((3, 1))
        self.omega_ref = np.zeros((3, 1))
        self.Kp_x = ctrl_cfg.gains.Kp_x
        self.Kd_x = ctrl_cfg.gains.Kd_x
        self.Kp_q = ctrl_cfg.gains.Kp_q
        self.Kd_q = ctrl_cfg.gains.Kd_q
        self.B_matrix = self.calc_B_matrix(env.model)
        self._inverse_mixer = np.linalg.pinv(self.B_matrix)
        self._groups = thruster_groups(env.model)
        self._bounds = actuator_bounds(env.model)

    calc_B_matrix = staticmethod(calc_B_matrix)

    def _apply_ctrl_constraint(self, env, u):
        return constrain_thrust(u, self._groups, np)

    def get_control_input(self, env):
        obs = env.get_obs()
        position, quaternion = self.planner.get_reference(obs)
        reference = np.concatenate((np.asarray(position).ravel(), np.asarray(quaternion).ravel()))
        return compute_control(
            obs[None, :], reference[None, :], self.v_ref,
            (self.Kp_x, self.Kd_x, self.Kp_q, self.Kd_q),
            self._inverse_mixer, self._groups,
            velocity_frame=env.env_cfg.sim.obs.v_frame, xp=np,
            mixer=self.B_matrix, bounds=self._bounds,
        )[0]

    def _log(self, run_id, timestamp, env):
        raise NotImplementedError(f"Logging is not implemented for {type(self).__name__}")
