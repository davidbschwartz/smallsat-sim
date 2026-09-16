"""Initial-condition sampling; reset clocks are episode-relative.

Masked resets preserve realized effect state and restart onset comparisons at zero.
Host randomization owns resampling effects between rollouts.
"""
import jax
import jax.numpy as jnp

def sample_initial_quaternion(rng: jnp.ndarray) -> jnp.ndarray:
    """
    Sample the legacy initial attitude distribution, preserving seeded runs.
    """
    key, subkey1, subkey2 = jax.random.split(rng, 3)
    theta1 = jax.random.uniform(key, (1,)) * 2 * jnp.pi
    theta2 = jax.random.uniform(subkey1, (1,)) * 2 * jnp.pi
    theta3 = jax.random.uniform(subkey2, (1,)) * 2 * jnp.pi

    w = jnp.sin(theta1) * jnp.cos(theta2) * jnp.cos(theta3) + jnp.cos(theta1) * jnp.sin(
        theta2
    ) * jnp.sin(theta3)
    x = jnp.cos(theta1) * jnp.sin(theta2) * jnp.cos(theta3) - jnp.sin(theta1) * jnp.cos(
        theta2
    ) * jnp.sin(theta3)
    y = jnp.sin(theta1) * jnp.cos(theta2) * jnp.cos(theta3) - jnp.cos(theta1) * jnp.sin(
        theta2
    ) * jnp.sin(theta3)
    z = jnp.cos(theta1) * jnp.cos(theta2) * jnp.sin(theta3) + jnp.sin(theta1) * jnp.sin(
        theta2
    ) * jnp.cos(theta3)

    quat = jnp.array([w, x, y, z]).reshape(-1)
    return quat / jnp.linalg.norm(quat)


def sample_initial_conditions(key, initial_position, initial_velocity, num_envs,
                              max_offset, max_linear_velocity, max_angular_velocity):
    """Sample each row with a stable key split order; linear velocity is world-frame."""
    key, sample_key = jax.random.split(key)
    row_keys = jax.random.split(sample_key, num_envs)

    def sample_row(row_key):
        position_key, attitude_key, linear_key, angular_key = jax.random.split(row_key, 4)
        uniform = jax.random.uniform(position_key, (2,))
        radius = max_offset * jnp.sqrt(uniform[0])
        angle = 2 * jnp.pi * uniform[1]
        position = initial_position[:3] + jnp.array([
            radius * jnp.cos(angle), radius * jnp.sin(angle), 0.])
        quaternion = sample_initial_quaternion(attitude_key).astype(initial_position.dtype)
        linear = jax.random.uniform(linear_key, (3,),
            minval=-float(max_linear_velocity), maxval=float(max_linear_velocity))
        angular = jax.random.uniform(angular_key, (3,),
            minval=-float(max_angular_velocity), maxval=float(max_angular_velocity))
        velocity = initial_velocity.at[:3].set(linear).at[3:6].set(angular)
        return jnp.concatenate((position, quaternion)), velocity

    positions, velocities = jax.vmap(sample_row)(row_keys)
    return key, positions, velocities


def validate_reset_mask(mask, num_envs):
    mask = jnp.asarray(mask, dtype=bool)
    if mask.shape != (num_envs,):
        raise ValueError(f"reset_mask must have shape ({num_envs},), got {mask.shape}")
    return mask


def merge_reset_rows(initial, current, mask):
    """Select complete reset rows without recomputing any unselected row."""
    def merge(initial_leaf, current_leaf):
        if not hasattr(initial_leaf, 'shape') or not initial_leaf.ndim:
            return current_leaf
        if initial_leaf.shape[0] != mask.shape[0]:
            return current_leaf
        row_mask = mask.reshape(mask.shape + (1,) * (initial_leaf.ndim - 1))
        return jnp.where(row_mask, initial_leaf, current_leaf)
    return jax.tree.map(merge, initial, current)
