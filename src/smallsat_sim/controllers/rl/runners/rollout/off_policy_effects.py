"""Stable, per-environment fault realizations for independent episode resets."""
from dataclasses import replace
import jax.numpy as jnp
from smallsat_sim.envs.effects.actuator_kernels import BatchedFaultState
from smallsat_sim.envs.vec_env.reset import merge_reset_rows


def batched_effect_state(env, state):
    """Expand shared GP tables and fill inactive optional buffers before tracing."""
    faults = []
    for effect, fault in zip(env.perturbations, state.perturbation_states, strict=True):
        if not isinstance(fault, BatchedFaultState):
            faults.append(fault)
            continue
        if hasattr(effect, '_gp_num_points'):
            shape = (env.num_envs, env.act_dim, effect._gp_num_points)
            def table(value):
                return jnp.zeros(shape, jnp.float32) if value is None else jnp.broadcast_to(value, shape)
            fault = replace(fault, gp_x_samples=table(fault.gp_x_samples),
                            gp_y_samples=table(fault.gp_y_samples))
        if fault.failure_value == 2 and fault.max_thruster_force is None:
            fault = replace(fault, max_thruster_force=jnp.zeros_like(fault.start_times))
        faults.append(fault)
    return state.replace(perturbation_states=tuple(faults))


def merge_effects(candidate, current, done):
    """Merge row data, keeping global RNG keys intact even when num_envs == 2."""
    merged = merge_reset_rows(candidate, current, done)
    return tuple(replace(new, rng=old.rng) if hasattr(old, 'rng') else new
                 for new, old in zip(merged, current, strict=True))
