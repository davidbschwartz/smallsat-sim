"""Pure external-wrench composition in world coordinates."""
import jax.numpy as jnp
from .custom import CustomEffectState, apply_custom_effect
from .disturbances import constant_force_apply_from_state


def apply_world_wrenches(states, time, zero_wrench):
    """Evaluate each effect once per control interval and sum world force/torque."""
    wrench = zero_wrench
    updated_states = []
    for state in states:
        if isinstance(state, CustomEffectState):
            contribution, state = apply_custom_effect(state, jnp.zeros_like(wrench), time)
            wrench = wrench + contribution
        elif state is not None and state.params.get('const_force') is not None:
            contribution, state = constant_force_apply_from_state(
                state, time, state.params['const_force'])
            wrench = wrench + contribution
        updated_states.append(state)
    return wrench, tuple(updated_states)
