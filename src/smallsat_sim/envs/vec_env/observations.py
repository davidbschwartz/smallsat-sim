"""Task features: world position/error, attitude log error, body velocities."""
import jax
import jax.numpy as jnp
from smallsat_sim.utils.quaternions_jax import quaternion_log_error


def state_features(qpos, velocity_body, angular_velocity, reference):
    """Return [position error, attitude error, body velocity, angular velocity]."""
    reference = jnp.broadcast_to(jnp.asarray(reference), qpos.shape)
    position_error = qpos[:, :3] - reference[:, :3]
    attitude_error = jax.vmap(quaternion_log_error)(qpos[:, 3:7], reference[:, 3:7])
    return jnp.concatenate((position_error, attitude_error, velocity_body, angular_velocity), axis=1)


def body_velocity(batch):
    return jnp.einsum('bji,bj->bi', batch.xmat[:, 1], batch.qvel[:, :3])


def mjx_state_features(batch, reference):
    return state_features(batch.qpos, body_velocity(batch), batch.qvel[:, 3:6], reference)


def mjx_observations(batch):
    return jnp.concatenate((batch.qpos, body_velocity(batch), batch.qvel[:, 3:6]), axis=1)
