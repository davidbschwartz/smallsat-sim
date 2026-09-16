"""Resolve independent Astrobee environment and on-policy training settings."""

from smallsat_sim.planners.ad_star.config import planner_settings

import numpy as np
from smallsat_sim.envs.effects.custom import validate_custom_effects
from smallsat_sim.envs.config import load_env_settings
from smallsat_sim.envs.effects.catalog import FAULT_NAMES, fault_weights
from smallsat_sim.controllers.rl.config import load_training_config, policy_context, validate_sac_config
from smallsat_sim.configuration import (
    settings, apply_overrides, flatten_overrides, resolve_algorithm,
)


def resolve_config(
    *,
    seed=None,
    algorithm=None,
    overrides=None,
    init_pos=None,
    max_start_offset=None,
    train_with_failures=None,
    use_pretrained=None,
    use_adaptive_approach=None,
    am_architecture=None,
    adaptive_context_mode=None,
    num_envs=None,
    failure_fraction=None,
):
    """Resolve independent settings before allocating an environment.

    Nested and dotted RL overrides have identical meanings. Named options take
    precedence, except conflicting algorithm selections are rejected.
    """
    config = load_env_settings("astrobee_rl")
    config.planner = planner_settings(start=(0, 0, 10), goal=(0, 3, -10))
    config.training = load_training_config()
    overrides = normalize_overrides(overrides, config)
    algorithm = resolve_algorithm(
        algorithm,
        {"algorithm": overrides.get("training.algorithm")},
        config.training.algorithm,
    )
    if algorithm == 'sac':
        config.training.use_adaptive_approach = False
    apply_overrides(config, overrides)
    if seed is not None:
        config.sim.seed = seed
    if algorithm is not None:
        config.training.algorithm = algorithm
    if init_pos is not None:
        config.Bodies.bodies_list[0].pos = init_pos
    if max_start_offset is not None:
        config.Bodies.max_start_offset = max_start_offset
    if train_with_failures is not None:
        config.environment.train_with_failures = train_with_failures
    if use_pretrained is not None:
        config.training.use_pretrained = use_pretrained
    if use_adaptive_approach is not None:
        config.training.use_adaptive_approach = use_adaptive_approach
    if am_architecture is not None:
        config.training.am_architecture = am_architecture
    if adaptive_context_mode is not None:
        config.training.adaptive_context_mode = adaptive_context_mode
    if num_envs is not None:
        config.environment.num_envs = int(num_envs)
    if failure_fraction is not None:
        config.environment.failure_fraction = float(failure_fraction)
    validate_custom_effects(config.environment.custom_faults)
    validate_custom_effects(config.environment.custom_disturbances)
    fault_weights(config.environment.failure_distribution)
    training = config.pop("training")
    if training.algorithm == 'sac':
        validate_sac_config(training, config.environment.num_envs)
    # Only these policy choices determine environment allocation and observation shape.
    config.context = policy_context(training)
    config.max_episode_len = int(getattr(training, training.algorithm.upper()).max_ep_len)
    config.run_id = training.rl_run_id
    return settings({"env": config, "training": training})


def normalize_overrides(values, config):
    """Translate old RL paths at the input boundary; runtime uses domain sections."""
    result = {}
    for path, value in flatten_overrides(values or {}).items():
        original = path
        if path.startswith("RL."):
            path = path[3:]
        field = path.split(".", 1)[0]
        if field in config.environment:
            path = f"environment.{path}"
        elif field in config.training:
            path = f"training.{path}"
        elif field == "PD":
            path = f"control.{path}"
        if path in result:
            raise ValueError(f"Duplicate override path: {original} resolves to {path}")
        if path == "environment.failure_distribution" and isinstance(value, (list, tuple)):
            # Existing checkpoint metadata stores weights in the original category order.
            if len(value) != len(FAULT_NAMES):
                raise ValueError("Legacy failure_distribution must contain exactly five weights")
            value = dict(zip(FAULT_NAMES, value))
            fault_weights(value)
        result[path] = value
    return result
