"""Sample and schedule per-environment effects for vectorized rollouts."""

import jax
import jax.numpy as jnp
from .custom import CustomEffectState
from .catalog import FAULT_NAMES, FAULTS_BY_NAME, FaultSpec, fault_weights, validate_fault
from .disturbances import DisturbanceStatus
from .actuator_kernels import PerturbationStatus


def schedule_fault(env, spec: FaultSpec, *, key):
    """Record targets and onset in fault state; stepping later transforms commands."""
    index = validate_fault(spec, env.model_cfg, env.num_envs)
    ids = jnp.asarray(range(env.num_envs) if spec.env_ids is None else spec.env_ids, dtype=jnp.int32)
    if ids.size == 0:
        return
    if index is not None and bool(jnp.any(
        env.thruster_occupancy.mask[ids, index] != PerturbationStatus.OPERATIONAL.value
    )):
        raise ValueError("Requested actuator already has a scheduled fault")
    params = ({"valve_min": spec.minimum, "valve_max": spec.maximum}
              if FAULTS_BY_NAME[spec.effect].accepts_bounds else {})
    env.perturbations.named(spec.effect).activate(
        key, ids, index, start_time=spec.start_time, **params
    )
    _publish_states(env)


def schedule_random_faults(
    env,
    key,
    fraction_perturbed_envs: float,
    perturbation_distribution=None,
    start_time: float | None = None,
) -> None:
    """
    Schedule one sampled actuator fault per selected environment.
    The thrusters are picked at random. ``start_time`` controls when the
    sampled failures become active in simulation time.
    """
    if perturbation_distribution is None:
        perturbation_distribution = env.env_cfg.environment.failure_distribution
    weights = jnp.asarray(fault_weights(perturbation_distribution))

    # Calculate number of environments to perturb
    clamped_fraction = max(0.0, min(1.0, fraction_perturbed_envs))
    num_perturbed = int(env.num_envs * clamped_fraction)
    if num_perturbed == 0:
        return

    # Split the key for permutation, categorical sampling, and perturbation subkeys
    perm_key, cat_key, subkeys_key = jax.random.split(key, 3)
    subkeys = jax.random.split(subkeys_key, len(FAULT_NAMES))

    # Prefer environments without any active failure/disturbance.
    # If that pool is exhausted, prefer envs with disturbances only
    # before reusing already perturbed envs.
    has_perturbation, has_disturbance = _get_active_failure_masks(env)
    clean_envs = jnp.logical_not(jnp.logical_or(has_perturbation, has_disturbance))
    disturbed_only_envs = jnp.logical_and(
        has_disturbance, jnp.logical_not(has_perturbation)
    )
    selected_indices = _select_envs_with_priority(
        env,
        perm_key,
        num_perturbed,
        [clean_envs, disturbed_only_envs, has_perturbation],
    )

    # Sample per-env perturbation types without host-side partitioning
    total_weight = jnp.sum(weights)
    safe_dist = jnp.where(total_weight > 0, weights / total_weight, weights)
    logits = jnp.where(weights > 0, jnp.log(safe_dist + 1e-8), -jnp.inf)
    categories = jax.random.categorical(cat_key, logits, shape=(num_perturbed,))

    for category, name in enumerate(FAULT_NAMES):
        env.perturbations.named(name).activate(
            subkeys[category], selected_indices[categories == category],
            start_time=start_time,
        )
    _publish_states(env)


def schedule_random_disturbances(
    env,
    key,
    fraction_disturbed_envs: float,
    start_time: float | None = None,
) -> None:
    """
    Schedule constant wrenches for selected environments; stepping adds the forces.
    """
    if env.disturbances is None:
        return

    clamped_fraction = max(0.0, min(1.0, fraction_disturbed_envs))
    num_disturbed = int(env.num_envs * clamped_fraction)
    if num_disturbed == 0:
        return

    # Prefer environments without any active failure/disturbance.
    # If needed, reuse already disturbed envs before overlapping with perturbations.
    has_perturbation, has_disturbance = _get_active_failure_masks(env)
    clean_envs = jnp.logical_not(jnp.logical_or(has_perturbation, has_disturbance))
    perturb_only_envs = jnp.logical_and(
        has_perturbation, jnp.logical_not(has_disturbance)
    )
    selected_indices = _select_envs_with_priority(
        env,
        key,
        num_disturbed,
        [clean_envs, has_disturbance, perturb_only_envs],
    )

    constant_force_disturbance = _constant_force(env)

    if constant_force_disturbance is None:
        return

    constant_force_disturbance.const_force_disturbance(
        selected_indices,
        start_time=0.0 if start_time is None else start_time,
    )
    _publish_states(env)


def set_constant_wrench_disturbances(
    env,
    env_indices: jnp.ndarray,
    wrenches: jnp.ndarray,
    start_time: float = 0.0,
) -> None:
    """
    Set deterministic world-frame force/torque for selected environments.
    """
    if env.disturbances is None:
        return

    env_indices = jnp.asarray(env_indices, dtype=jnp.int32)
    if env_indices.size == 0:
        return

    constant_force_disturbance = _constant_force(env)

    if constant_force_disturbance is None:
        return

    wrenches = jnp.asarray(wrenches, dtype=jnp.float32)
    if wrenches.ndim != 2 or wrenches.shape[-1] != 6:
        raise ValueError("wrenches must have shape (num_selected_envs, 6).")
    if wrenches.shape[0] != env_indices.shape[0]:
        raise ValueError("env_indices and wrenches must have matching rows.")

    constant_force_disturbance.activate(
        env_indices, wrenches=wrenches, start_time=start_time
    )
    _publish_states(env)


def reset_and_randomize(env, key, *, enabled):
    cfg = env.env_cfg.environment
    env.reset()
    env.reset_perturbations()
    env.reset_disturbances()
    if not enabled:
        return
    for value in (cfg.failure_fraction, cfg.disturbance_fraction):
        if not 0 <= value <= 1:
            raise ValueError("Randomization fractions must be in [0, 1]")
    onset1, onset2, failure, disturbance = jax.random.split(key, 4)
    def onset(key, interval):
        low, high = interval
        if not 0 <= low <= high:
            raise ValueError("Onset interval must satisfy 0 <= low <= high")
        return float(jax.random.uniform(key, (), minval=low, maxval=high))
    env.schedule_random_faults(failure, cfg.failure_fraction,
        cfg.failure_distribution, start_time=onset(onset1, cfg.failure_start_time))
    env.schedule_random_disturbances(disturbance, cfg.disturbance_fraction,
        start_time=onset(onset2, cfg.disturbance_start_time))


def _publish_states(env):
    """Publish host-side changes to the next functional rollout snapshot."""
    env._refresh_effect_states()


def _constant_force(env):
    return next((effect for effect in env.disturbances.disturbances
                 if getattr(effect, "failure_type", None) == DisturbanceStatus.CONSTANT_FORCE), None)


def _get_active_failure_masks(env) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Return boolean masks `(has_perturbation, has_disturbance)` per environment.
    """
    has_perturbation = jnp.zeros((env.num_envs,), dtype=bool)
    has_disturbance = jnp.zeros((env.num_envs,), dtype=bool)

    occupancy = getattr(env, "thruster_occupancy", None)
    thruster_mask = occupancy.mask if occupancy is not None else None
    if thruster_mask is not None and thruster_mask.shape[0] == env.num_envs:
        has_perturbation = jnp.any(
            thruster_mask != PerturbationStatus.OPERATIONAL.value, axis=1
        )

    for effect_state in getattr(env, "perturbation_states", ()):
        if isinstance(effect_state, CustomEffectState):
            has_perturbation = has_perturbation | effect_state.active_mask

    disturbance_states = getattr(env, "disturbance_states", ()) or ()
    for state in disturbance_states:
        if state is None:
            continue
        active_mask = getattr(state, "active_mask", None)
        if active_mask is None:
            continue
        active_mask = jnp.asarray(active_mask).astype(bool)
        if active_mask.shape[0] == env.num_envs:
            has_disturbance = jnp.logical_or(has_disturbance, active_mask)

    return has_perturbation, has_disturbance


def _select_envs_with_priority(
    env,
    key: jnp.ndarray,
    num_selected: int,
    priority_masks: list[jnp.ndarray],
) -> jnp.ndarray:
    """
    Select `num_selected` unique env indices from priority tiers in order.
    """
    if num_selected <= 0:
        return jnp.array([], dtype=jnp.int32)
    remaining = min(num_selected, env.num_envs)
    permutation = jax.random.permutation(key, jnp.arange(env.num_envs, dtype=jnp.int32))
    selected = jnp.zeros((env.num_envs,), dtype=bool)

    for tier in priority_masks:
        if remaining <= 0:
            break
        tier = jnp.asarray(tier, dtype=bool)
        if tier.shape[0] != env.num_envs:
            continue
        candidates = permutation[(tier & ~selected)[permutation]]
        chosen = candidates[:remaining]
        selected = selected.at[chosen].set(True)
        remaining -= chosen.size

    if remaining > 0:
        chosen = permutation[~selected[permutation]][:remaining]
        selected = selected.at[chosen].set(True)
    return permutation[selected[permutation]]
