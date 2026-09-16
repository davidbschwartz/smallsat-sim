"""Continuous domain randomization through the existing custom-effect API."""

import jax
import jax.numpy as jnp


def sample_thrust(key, context, low, high):
    return {
        "scale": jax.random.uniform(
            key, (context.num_envs, len(context.vehicle.actuators)), minval=low, maxval=high
        )
    }


def apply_thrust(data, values, time, dt):
    return values * data["scale"], data


def sample_wrench(key, context, force, torque):
    limits = jnp.array([force] * 3 + [torque] * 3)
    return {
        "wrench": jax.random.uniform(key, (context.num_envs, 6), minval=-1.0, maxval=1.0) * limits
    }


def apply_wrench(data, values, time, dt):
    return values + data["wrench"], data


def training_effects(distribution):
    # Keep archived configuration fingerprints stable; the legacy package forwards here.
    prefix = "experiments.demonstration_use_cases.effects:"
    fault = {
        "sample": prefix + "sample_thrust",
        "apply": prefix + "apply_thrust",
        "version": "1",
        "probability": 1.0,
        "start_time": 0.0,
        "params": dict(zip(("low", "high"), distribution["thrust"])),
    }
    wrench = {
        "sample": prefix + "sample_wrench",
        "apply": prefix + "apply_wrench",
        "version": "1",
        "probability": 1.0,
        "start_time": 0.0,
        "params": {k: distribution[k] for k in ("force", "torque")},
    }
    return [fault], [wrench]
