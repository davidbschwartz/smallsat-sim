"""Numerical integration tests; no viewer, policy, or checkpoint required."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
import pytest

from smallsat_sim.envs.vec_env import freeflyer, mjx_backend
from smallsat_sim.envs.vec_env.observations import mjx_state_features
from smallsat_sim.utils.quaternions_jax import quaternion_to_rotation_matrix


@pytest.fixture(scope="module")
def free_body():
    # Site transmissions apply body-frame forces/torques, like the thruster mixer.
    gears = np.eye(6)
    actuators = "".join(
        f'<general site="thrust" gear="{" ".join(map(str, gear))}"/>'
        for gear in gears
    )
    model = mujoco.MjModel.from_xml_string(f"""
        <mujoco>
          <option timestep="0.002" gravity="0 0 0" integrator="Euler"/>
          <worldbody><body pos="0 0 1">
            <freejoint/>
            <inertial pos="0 0 0" mass="2" diaginertia="0.2 0.3 0.4"/>
            <site name="thrust"/>
          </body></worldbody>
          <actuator>{actuators}</actuator>
        </mujoco>
    """)
    mjx_model = mjx.put_model(model)
    data = mjx.make_data(mjx_model)
    batch = jax.tree.map(lambda x: jnp.stack([x, x]), data)
    config = SimpleNamespace(
        num_envs=2,
        init_qpos=batch.qpos,
        init_qvel=batch.qvel,
        max_start_offset=0.5,
        # Nonzero velocity with randomized attitude catches reset frame mismatches.
        max_start_linear_velocity=0.1,
        max_start_angular_velocity=0.2,
        thruster_mixer_T=jnp.eye(6),
        base_disturbance_states=(),
        base_perturbation_states=(),
        model_dt=model.opt.timestep,
        mass=2.0,
        inertia_diag=jnp.array([0.2, 0.3, 0.4]),
    )
    key = jax.random.PRNGKey(17)
    state = jax.jit(lambda key: mjx_backend.reset_from_key(
        key,
        mjx_model=mjx_model,
        init_qpos=config.init_qpos,
        init_qvel=config.init_qvel,
        num_envs=config.num_envs,
        max_start_offset=config.max_start_offset,
        max_start_linear_velocity=config.max_start_linear_velocity,
        max_start_angular_velocity=config.max_start_angular_velocity,
    ))(key)
    return mjx_model, config, key, state


def test_freeflyer_reset_matches_mjx(free_body):
    _, config, key, state = free_body
    actual = freeflyer.reset_from_key(key, config=config)
    expected = freeflyer.from_mjx(state)
    for field in ("qpos", "vel_body", "omega", "time", "ctrl", "actuator_force",
                  "terminal_hold_counts", "rng"):
        np.testing.assert_allclose(
            getattr(actual, field), getattr(expected, field), rtol=0, atol=1e-6,
            err_msg=field,
        )
    reference = config.init_qpos[0]
    np.testing.assert_allclose(
        freeflyer.state_features(actual, reference),
        mjx_state_features(state.mjx_batch, reference),
        rtol=0, atol=1e-6,
    )


def test_freeflyer_short_rollout_matches_mjx(free_body):
    model, config, _, state = free_body
    # Nonzero world velocity and randomized attitude expose frame-conversion errors.
    batch = state.mjx_batch.replace(
        qvel=state.mjx_batch.qvel.at[:, :3].set(jnp.array([0.1, -0.2, 0.05]))
    )
    compact = freeflyer.from_mjx(state.replace(mjx_batch=batch))
    controls = jax.random.uniform(
        jax.random.PRNGKey(23), (16, 2, 6), minval=-0.2, maxval=0.2,
    )

    @jax.jit
    def rollout(batch, compact):
        def step(carry, ctrl):
            batch, compact = carry
            batch = jax.vmap(mjx.step, in_axes=(None, 0))(
                model, batch.replace(ctrl=ctrl)
            )
            compact = freeflyer.integrate_substep(
                compact, ctrl, jnp.zeros((2, 6)), config
            )
            return (batch, compact), (batch.qpos, batch.qvel, compact)
        return jax.lax.scan(step, (batch, compact), controls)[1]

    qpos, qvel, compact = rollout(batch, compact)
    # Explicit vs semi-implicit Euler differs by O(dt^2) per step. Over 32 ms,
    # allow 0.1 mm position, 0.2 mrad attitude, and 0.2 mm/s or mrad/s velocity.
    np.testing.assert_allclose(compact.qpos[..., :3], qpos[..., :3], rtol=0, atol=1e-4)
    quat_error = np.minimum(
        np.linalg.norm(compact.qpos[..., 3:7] - qpos[..., 3:7], axis=-1),
        np.linalg.norm(compact.qpos[..., 3:7] + qpos[..., 3:7], axis=-1),
    )
    assert np.all(quat_error < 1e-4), quat_error
    # Rotate MJX world-frame linear velocity into the current body frame.
    rotations = jax.vmap(quaternion_to_rotation_matrix)(qpos[..., 3:7])
    body_velocity = jnp.einsum("tnji,tnj->tni", rotations, qvel[..., :3])
    np.testing.assert_allclose(compact.vel_body, body_velocity, rtol=0, atol=2e-4)
    np.testing.assert_allclose(compact.omega, qvel[..., 3:6], rtol=0, atol=2e-4)


def test_freeflyer_actuator_limits_match_applied_force(free_body):
    _, config, _, state = free_body
    compact = freeflyer.from_mjx(state)
    limited = SimpleNamespace(**vars(config),
        actuator_control_limits=jnp.tile(jnp.array([-.1, .1]), (6, 1)),
        actuator_force_limits=jnp.tile(jnp.array([-.05, .05]), (6, 1)))
    controls = jnp.full((2, 6), .3)
    actual = freeflyer.integrate_substep(compact, controls, jnp.zeros((2, 6)), limited)
    expected = freeflyer.integrate_substep(compact, jnp.full((2, 6), .05), jnp.zeros((2, 6)), config)
    np.testing.assert_allclose(actual.vel_body, expected.vel_body)
    np.testing.assert_allclose(actual.omega, expected.omega)
    np.testing.assert_allclose(actual.ctrl, controls)
    np.testing.assert_allclose(actual.actuator_force, .05)


def test_freeflyer_rejects_unsupported_physics():
    from smallsat_sim.envs.vec_env.config import validate_physics
    model = mujoco.MjModel.from_xml_string('''<mujoco><option gravity="0 0 0"/>
      <worldbody><body><freejoint/><geom type="sphere" size=".1" mass="1"/>
      </body></worldbody></mujoco>''')
    validate_physics(model, 'freeflyer')
    model.opt.gravity[2] = -9.81
    with pytest.raises(ValueError, match='passive forces'):
        validate_physics(model, 'freeflyer')
    validate_physics(model, 'mjx')


def test_freeflyer_longer_rollout_with_world_torque(free_body):
    from smallsat_sim.envs.vec_env.mjx_backend import advance_physics
    model, config, _, initial = free_body
    compact = freeflyer.from_mjx(initial)
    world_wrench = jnp.tile(jnp.array([.01, -.01, .01, .005, -.004, .003]), (2, 1))
    controls = jnp.full((2, 6), .01)

    @jax.jit
    def rollout(batch, compact):
        def interval(carry, _):
            batch, compact = carry
            torque_body = jnp.einsum('bji,bj->bi', batch.xmat[:, 1], world_wrench[:, 3:])
            batch = advance_physics(model, batch.replace(ctrl=controls,
                qfrc_applied=world_wrench.at[:, 3:].set(torque_body)), 3)
            compact = jax.lax.fori_loop(0, 3, lambda _, current:
                freeflyer.integrate_substep(current, controls, world_wrench, config), compact)
            return (batch, compact), None
        return jax.lax.scan(interval, (batch, compact), None, length=100)[0]

    batch, actual = rollout(initial.mjx_batch, compact)
    expected = freeflyer.from_mjx(initial.replace(mjx_batch=batch))
    # Different Euler schemes over 0.6 seconds: bound position and body-velocity drift.
    np.testing.assert_allclose(actual.qpos[:, :3], expected.qpos[:, :3], atol=5e-4, rtol=0)
    np.testing.assert_allclose(actual.vel_body, expected.vel_body, atol=5e-4, rtol=0)
    np.testing.assert_allclose(actual.omega, expected.omega, atol=5e-4, rtol=0)
    quaternion_distance = np.minimum(np.linalg.norm(actual.qpos[:, 3:] - expected.qpos[:, 3:], axis=1),
                                     np.linalg.norm(actual.qpos[:, 3:] + expected.qpos[:, 3:], axis=1))
    assert np.all(quaternion_distance < 5e-4)
