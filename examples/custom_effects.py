"""Importable examples for environment.custom_faults/custom_disturbances."""
import jax
import jax.numpy as jnp


def sample_motor_loss(key, context, actuator, minimum=0.2, maximum=0.8):
    names = [item.name for item in context.vehicle.actuators]
    if actuator not in names:
        raise ValueError(f"Unknown actuator {actuator!r}")
    if not 0 <= minimum <= maximum <= 1:
        raise ValueError("Motor efficiency bounds must satisfy 0 <= minimum <= maximum <= 1")
    efficiency = jnp.ones((context.num_envs, len(names)))
    efficiency = efficiency.at[:, names.index(actuator)].set(
        jax.random.uniform(key, (context.num_envs,), minval=minimum, maxval=maximum))
    return {"efficiency": efficiency, "elapsed": jnp.zeros((context.num_envs,))}


def apply_motor_loss(state, controls, time, dt):
    return controls * state["efficiency"], {**state, "elapsed": state["elapsed"] + dt}


def sample_wrench(key, context, wrench):
    wrench = jnp.asarray(wrench, dtype=jnp.float32)
    if wrench.shape != (6,):
        raise ValueError("wrench must contain world-frame force xyz and torque xyz")
    return {"wrench": jnp.broadcast_to(wrench, (context.num_envs, 6))}


def apply_wrench(state, zero_wrench, time, dt):
    return state["wrench"], state
