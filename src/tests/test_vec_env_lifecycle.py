"""State/reset contracts independent of policy training."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
import pytest

from smallsat_sim.envs.vec_env import freeflyer, mjx_backend

from smallsat_sim.envs.vec_env.runtime import VecEnv
from smallsat_sim.envs.vec_env.types import VecEnvState


@pytest.fixture
def physics():
    model = mjx.put_model(mujoco.MjModel.from_xml_string('''
      <mujoco><option gravity="0 0 0"/><worldbody><body>
      <freejoint/><geom type="sphere" size=".1" mass="1"/>
      </body></worldbody></mujoco>'''))
    data = mjx.make_data(model)
    batch = jax.tree.map(lambda leaf: jnp.stack([leaf, leaf]), data)
    config = SimpleNamespace(mjx_model=model, init_qpos=batch.qpos, init_qvel=batch.qvel,
        num_envs=2, max_start_offset=.1, max_start_linear_velocity=0.,
        max_start_angular_velocity=0.)
    return config, VecEnvState(jax.random.PRNGKey(4), batch, jnp.zeros(2, jnp.int32))


def test_snapshot_reads_current_runtime_fields(physics):
    _, initial = physics
    env = VecEnv.__new__(VecEnv)
    # A legacy cached snapshot must not override current runtime fields.
    env._state = initial
    env._rng = initial.rng
    env.mjx_batch = initial.mjx_batch.replace(time=jnp.ones(2))
    env._terminal_hold_counts = jnp.array([2, 3])
    env.next_rng_keys()
    snapshot = env.state_struct
    np.testing.assert_array_equal(snapshot.rng, env._rng)
    np.testing.assert_array_equal(snapshot.mjx_batch.time, [1., 1.])
    np.testing.assert_array_equal(snapshot.terminal_hold_counts, [2, 3])


def test_reset_clears_dirty_physics(physics):
    config, initial = physics
    dirty = initial.mjx_batch.replace(time=jnp.ones(2) * 7,
        qfrc_applied=jnp.ones((2, 6)), qacc_warmstart=jnp.ones((2, 6)))
    reset = mjx_backend.reset(initial.replace(mjx_batch=dirty), config, jnp.ones(2, dtype=bool))
    np.testing.assert_array_equal(reset.mjx_batch.time, [0., 0.])
    np.testing.assert_array_equal(reset.mjx_batch.qfrc_applied, np.zeros((2, 6)))
    np.testing.assert_array_equal(reset.mjx_batch.qacc_warmstart, np.zeros((2, 6)))


def test_masked_reset_preserves_every_unselected_leaf(physics):
    config, initial = physics
    state = initial.replace(mjx_batch=initial.mjx_batch.replace(
        time=jnp.array([3., 4.]), qfrc_applied=jnp.ones((2, 6))),
        terminal_hold_counts=jnp.array([2, 3]))
    actual = jax.jit(lambda state: mjx_backend.reset(state, config,
        jnp.array([True, False])))(state)
    assert actual.mjx_batch.time[0] == 0
    for before, after in zip(jax.tree.leaves(state.mjx_batch),
                             jax.tree.leaves(actual.mjx_batch), strict=True):
        if before.ndim and before.shape[0] == 2:
            np.testing.assert_array_equal(before[1], after[1])
    np.testing.assert_array_equal(actual.terminal_hold_counts, [0, 3])


@pytest.mark.parametrize('mask', [[False, False], [True, True]])
def test_reset_mask_extremes_and_rng(physics, mask):
    config, state = physics
    config.thruster_mixer_T = jnp.zeros((0, 6))
    config.base_disturbance_states = config.base_perturbation_states = ()
    compact = freeflyer.from_mjx(state)
    actual = mjx_backend.reset(state, config, jnp.array(mask))
    compact_reset = freeflyer.reset_masked(compact, config, jnp.array(mask))
    np.testing.assert_array_equal(actual.rng, compact_reset.rng)
    np.testing.assert_array_equal(actual.rng, jax.random.split(state.rng)[0])
    if not any(mask):
        for before, after in zip(jax.tree.leaves(state.mjx_batch), jax.tree.leaves(actual.mjx_batch), strict=True):
            np.testing.assert_array_equal(before, after)


def test_world_torque_and_per_row_onset(physics):
    from smallsat_sim.envs.effects.disturbances import DisturbanceState
    config, state = physics
    # A quarter turn about Z maps world X torque to negative body Y.
    quaternion = jnp.array([2**-.5, 0., 0., 2**-.5])
    batch = state.mjx_batch.replace(qpos=state.mjx_batch.qpos.at[:, 3:].set(quaternion),
                                   time=jnp.array([0., 2.]))
    batch = jax.vmap(mjx.forward, in_axes=(None, 0))(config.mjx_model, batch)
    effect = DisturbanceState(state.rng, jnp.ones(2, bool), jnp.ones(2),
        {'const_force': jnp.tile(jnp.array([1., 0., 0., 1., 0., 0.]), (2, 1))})
    state = state.replace(mjx_batch=batch, disturbance_states=(effect,))
    _, _, generalized = jax.jit(mjx_backend.prepare_effects)(state, jnp.zeros((2, 0)))
    _, _, world = jax.jit(freeflyer.prepare_effects)(freeflyer.from_mjx(state), jnp.zeros((2, 0)))
    np.testing.assert_allclose(generalized, [[0]*6, [1, 0, 0, 0, -1, 0]], atol=2e-6)
    np.testing.assert_allclose(world, [[0]*6, [1, 0, 0, 1, 0, 0]], atol=2e-6)


def test_quaternion_helpers_preserve_scalar_batch_and_sign_contracts():
    from smallsat_sim.utils.quaternions_jax import (
        quaternion_to_rotation_matrix, quaternion_rate_matrix, quaternion_log_error)
    quaternion = jnp.array([.5, .5, .5, .5])
    np.testing.assert_allclose(quaternion_to_rotation_matrix(quaternion),
                              [[0, 0, 1], [1, 0, 0], [0, 1, 0]], atol=1e-6)
    for operation in (quaternion_to_rotation_matrix, quaternion_rate_matrix):
        np.testing.assert_allclose(operation(jnp.stack([quaternion, quaternion])),
                                  jnp.stack([operation(quaternion)] * 2))
    np.testing.assert_allclose(quaternion_log_error(quaternion, -quaternion), [0, 0, 0], atol=1e-6)
    for angle in (1e-7, np.pi-1e-6):
        target = jnp.array([np.cos(angle/2), np.sin(angle/2), 0, 0])
        np.testing.assert_allclose(quaternion_log_error(jnp.array([1., 0, 0, 0]), target),
                                  [angle, 0, 0], atol=1e-6)
