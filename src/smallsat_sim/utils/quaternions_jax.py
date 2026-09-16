"""Scalar-first quaternion algebra. Rotation matrices map body to world."""
import jax.numpy as jnp

def quaternion_log_error(
    q: jnp.ndarray, q_des: jnp.ndarray, eps: float = 1e-9
) -> jnp.ndarray:
    """
    SO(3) log-map (rotation vector) that rotates q -> q_des.
    """

    def _unit(a):
        return a / (jnp.linalg.norm(a) + eps)

    q = _unit(q)
    q_des = _unit(q_des)

    w, x, y, z = q
    qc = jnp.array([w, -x, -y, -z])
    w2, x2, y2, z2 = q_des
    we = w2 * qc[0] - x2 * qc[1] - y2 * qc[2] - z2 * qc[3]
    ex = w2 * qc[1] + x2 * qc[0] + y2 * qc[3] - z2 * qc[2]
    ey = w2 * qc[2] - x2 * qc[3] + y2 * qc[0] + z2 * qc[1]
    ez = w2 * qc[3] + x2 * qc[2] - y2 * qc[1] + z2 * qc[0]
    e = jnp.stack([ex, ey, ez])

    sign = jnp.where(we < 0.0, -1.0, 1.0)
    we = sign * we
    e = sign * e

    e_norm = jnp.linalg.norm(e)
    we_abs = jnp.clip(jnp.abs(we), 0.0, 1.0)
    theta = 2.0 * jnp.arctan2(e_norm, we_abs)
    axis = e / (e_norm + eps)
    logvec = axis * theta

    return logvec


def quaternion_to_rotation_matrix(q: jnp.ndarray, *, normalize: bool = True) -> jnp.ndarray:
    """Map (..., 4) scalar-first quaternions to (..., 3, 3) body-to-world rotations."""
    if normalize:
        q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-9)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return jnp.stack(
        [
            jnp.stack(
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                axis=-1,
            ),
            jnp.stack(
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                axis=-1,
            ),
            jnp.stack(
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                axis=-1,
            ),
        ],
        axis=-2,
    )


def quaternion_rate_matrix(q: jnp.ndarray, *, normalize: bool = True) -> jnp.ndarray:
    """Return (..., 4, 3) matrices mapping body angular velocity to quaternion rate."""
    if normalize:
        q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-9)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return 0.5 * jnp.stack(
        [
            jnp.stack([-x, -y, -z], axis=-1),
            jnp.stack([w, -z, y], axis=-1),
            jnp.stack([z, w, -x], axis=-1),
            jnp.stack([-y, x, w], axis=-1),
        ],
        axis=-2,
    )


def quat_multiply(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """q = quat_multiply(q1,q2) computes the quaternion product q of
    two quaternions q1 and q2.
    """
    if len(q1) == 4 and len(q2) == 4:
        eta1 = q1[0]
        eps1_1 = q1[1]
        eps1_2 = q1[2]
        eps1_3 = q1[3]

        eta2 = q2[0]
        eps2_1 = q2[1]
        eps2_2 = q2[2]
        eps2_3 = q2[3]

        q = jnp.array(
            [
                eta1 * eta2 - eps1_1 * eps2_1 - eps1_2 * eps2_2 - eps1_3 * eps2_3,
                eta1 * eps2_1 + eps1_1 * eta2 + eps1_2 * eps2_3 - eps1_3 * eps2_2,
                eta1 * eps2_2 - eps1_1 * eps2_3 + eps1_2 * eta2 + eps1_3 * eps2_1,
                eta1 * eps2_3 + eps1_1 * eps2_2 - eps1_2 * eps2_1 + eps1_3 * eta2,
            ]
        )

    else:
        raise ValueError("input must be of dim. 4 (unit quaternion)")
    return q


def quat_conjugate(q) -> jnp.ndarray:
    """q_conj = quat_conjugate(q) computes the quaternion conjugate
    q_conj of a quaternion q.
    """
    if len(q) == 4:
        q_conj = jnp.array([q[0], -q[1], -q[2], -q[3]])
    else:
        raise ValueError("input must be of dim. 4 (unit quaternion)")
    return q_conj

