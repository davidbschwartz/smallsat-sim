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

import jax
import jax.numpy as jnp

from smallsat_sim.controllers.pd.controller import PDController
from smallsat_sim.controllers.pd.common import compute_control, constrain_thrust


class VectorizedPDController(PDController):
    """Batched JAX adapter; VecEnv observations contain body-frame velocity."""

    def __init__(self, env, planner):
        super().__init__(env, planner)
        self._inverse_mixer = jnp.asarray(self._inverse_mixer)
        self._compute_control = jax.jit(lambda obs, reference, v_ref, gains: compute_control(
            obs, reference, v_ref, gains, self._inverse_mixer, self._groups,
            velocity_frame="body", xp=jnp, mixer=self.B_matrix, bounds=self._bounds,
        ))

    def _apply_ctrl_constraint(self, env, u):
        return constrain_thrust(u, self._groups, jnp)

    def get_control_input(self, env, next_waypoint=None):
        obs = env.get_obs()
        if next_waypoint is None:
            # Preserve the existing training fallback: hold the initial position.
            next_waypoint = jnp.asarray(env.env_cfg.Bodies.bodies_list[0].pos)
        reference = jnp.asarray(next_waypoint)
        if reference.ndim == 1:
            reference = jnp.broadcast_to(reference, (obs.shape[0], reference.shape[0]))
        if reference.shape[-1] == 3:
            quaternion = jnp.broadcast_to(jnp.array([1., 0., 0., 0.]), (obs.shape[0], 4))
            reference = jnp.concatenate((reference, quaternion), axis=-1)
        if reference.shape != (obs.shape[0], 7):
            raise ValueError("PD waypoint must have shape (3,), (7,), (num_envs, 3), or (num_envs, 7)")
        return self._compute_control(
            obs, reference, jnp.asarray(self.v_ref),
            (self.Kp_x, self.Kd_x, self.Kp_q, self.Kd_q),
        )
