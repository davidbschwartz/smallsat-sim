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

"""Shared batched PD math for NumPy and JAX environment adapters."""

import numpy as np


def calc_B_matrix(model):
    """Build the site-actuator mixer using each actuator's transmission target."""
    matrix = np.empty((6, model.nu))
    for i in range(model.nu):
        actuator = model.actuator(i)
        site_id = int(actuator.trnid[0])
        position = model.site(site_id if site_id >= 0 else i).pos
        matrix[:, i] = actuator.gear + np.concatenate(
            (np.zeros(3), np.cross(position, actuator.gear[:3]))
        )
    return matrix


def thruster_groups(model):
    # Preserve the existing axis-aligned thruster allocation policy without
    # assuming equal group sizes or consecutive actuator indices.
    return tuple(np.flatnonzero(model.actuator_gear[:, axis]) for axis in range(3))


def constrain_thrust(u, groups, xp):
    for group in groups:
        if len(group):
            shift = xp.minimum(xp.min(u[..., group], axis=-1, keepdims=True), 0)
            if xp is np:
                u[..., group] -= shift
            else:
                u = u.at[..., group].add(-shift)
    return u


def rotation(q, xp):
    w, x, y, z = (q[..., i] for i in range(4))
    return xp.stack((
        xp.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)), axis=-1),
        xp.stack((2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)), axis=-1),
        xp.stack((2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), axis=-1),
    ), axis=-2)


def compute_control(obs, reference, v_ref, gains, inverse_mixer, groups,
                    *, velocity_frame, xp, mixer=None, bounds=None):
    """Positions/v_ref are world-frame; angular velocities are body-frame.

    Retains the existing quaternion feedback and axis-group allocation law.
    Inputs have a leading environment dimension.
    """
    rot = rotation(obs[:, 3:7], xp)
    to_body = lambda v: xp.einsum("nji,nj->ni", rot, v)
    position_error = to_body(reference[:, :3] - obs[:, :3])
    target_velocity = xp.broadcast_to(xp.asarray(v_ref).reshape(1, 3), (len(obs), 3))
    if velocity_frame == "body":
        velocity_error = to_body(target_velocity) - obs[:, 7:10]
    elif velocity_frame == "inertial":
        velocity_error = to_body(target_velocity - obs[:, 7:10])
    else:
        raise ValueError(f"Unsupported PD velocity frame: {velocity_frame}")
    desired, current = reference[:, 3:7], obs[:, 3:7]
    # conjugate(current) * desired: attitude error expressed in body axes.
    eta = xp.sum(desired * current, axis=-1, keepdims=True)
    eps = (current[:, :1] * desired[:, 1:] - desired[:, :1] * current[:, 1:]
           + xp.cross(desired[:, 1:], current[:, 1:]))
    kp_x, kd_x, kp_q, kd_q = gains
    linear = kp_x * position_error + kd_x * velocity_error
    angular = kp_q * xp.where(eta >= 0, 1, -1) * eps - kd_q * obs[:, 10:13]
    wrench = xp.concatenate((linear, angular), axis=-1)
    raw = wrench @ inverse_mixer.T
    if mixer is None:
        return constrain_thrust(raw, groups, xp)
    return allocate_wrench(wrench, raw, mixer, groups, bounds, xp)


def allocate_wrench(wrench, raw, mixer, groups, bounds, xp):
    """Bounded least-squares allocation, with an exact balanced-layout fast path.

    Normalize each wrench axis to avoid sacrificing torque merely because its
    numerical units are smaller. Projected iterations support both NumPy and JAX.
    """
    lower, upper = (xp.asarray(value) for value in bounds)
    mixer = xp.asarray(mixer)
    shifted = constrain_thrust(raw, groups, xp)
    candidate = xp.clip(shifted, lower, upper)
    scale = xp.maximum(xp.linalg.norm(mixer, axis=1), 1e-12)
    matrix = mixer / scale[:, None]
    target = wrench / scale
    exact = xp.max(xp.abs(candidate @ matrix.T - target), axis=-1, keepdims=True) < 1e-7
    # Frobenius norm bounds the gradient Lipschitz constant.
    step = 1.0 / xp.maximum(xp.sum(matrix * matrix), 1e-12)
    def iteration(_, command):
        return xp.clip(command - step * ((command @ matrix.T - target) @ matrix), lower, upper)
    if xp is np:
        if np.all(exact):
            return candidate
        result = candidate.copy()
        for i in range(128):
            result = iteration(i, result)
    else:
        import jax
        result = jax.lax.fori_loop(0, 128, iteration, candidate)
    return xp.where(exact, candidate, result)
