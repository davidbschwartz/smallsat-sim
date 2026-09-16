"""Load training defaults and describe the policy context required by an environment."""

from copy import deepcopy

import numpy as np
import jax

from smallsat_sim.configuration import load_settings, settings



def load_training_config():
    return load_settings("training/on_policy.yaml")


def policy_context(config):
    """The allocation and normalization inputs shared by environment and policy."""
    return settings({
        "enabled": config.use_adaptive_approach,
        "mode": config.adaptive_context_mode,
        "history_len": config.context_window_len,
        "scale": deepcopy(config.context_scale),
    })


def validate_policy_context(env_config, training_config):
    expected = policy_context(training_config)
    for name, value in expected.items():
        if not np.array_equal(env_config.context[name], value):
            raise ValueError(
                f"Policy context differs at {name}; resolve settings together "
                "before constructing the environment"
            )


def validate_sac_config(config, num_envs):
    if config.algorithm != 'sac':
        raise ValueError('OffPolicyRunner requires algorithm sac')
    if config.use_adaptive_approach or config.use_pretrained:
        raise ValueError('SAC adaptation and pretraining are not supported')
    hp = config.SAC
    for name in ('total_transitions', 'collection_steps', 'replay_capacity', 'batch_size',
                 'learning_starts', 'updates_per_collection', 'max_ep_len', 'randomization_pool_size'):
        value = getattr(hp, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f'SAC {name} must be a positive integer')
    if not isinstance(hp.random_steps, int) or hp.random_steps < 0:
        raise ValueError('SAC random_steps must be a nonnegative integer')
    if hp.replay_capacity < hp.collection_steps * num_envs:
        raise ValueError('SAC replay capacity must hold a collection chunk')
    if not 0 <= hp.gamma <= 1 or not 0 < hp.tau <= 1:
        raise ValueError('SAC gamma must be in [0, 1] and tau in (0, 1]')
    for name in ('actor_lr', 'critic_lr', 'alpha_lr', 'initial_alpha'):
        if not np.isfinite(getattr(hp, name)) or getattr(hp, name) <= 0:
            raise ValueError(f'SAC {name} must be finite and positive')
    if not np.isfinite([hp.log_std_min, hp.log_std_max]).all() or hp.log_std_min >= hp.log_std_max:
        raise ValueError('SAC log standard deviation bounds must be finite and ordered')
    if hp.target_entropy is not None and not np.isfinite(hp.target_entropy):
        raise ValueError('SAC target entropy must be finite')
    if config.training_checkpoint_interval < 1:
        raise ValueError('Checkpoint interval must be positive')
    if not config.policy_hidden_sizes or any(x <= 0 for x in config.policy_hidden_sizes):
        raise ValueError('Policy hidden sizes must be positive')


def config_dict(value):
    """Resolve class-backed config values, including inherited defaults."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [config_dict(v) for v in value]
    if isinstance(value, (np.ndarray, jax.Array)):
        return np.asarray(value).tolist()
    if isinstance(value, dict):
        return {str(k): config_dict(v) for k, v in value.items()}
    return {
        name: config_dict(getattr(value, name))
        for name in dir(value)
        if not name.startswith("_")
        and (
            isinstance(getattr(value, name), type) or not callable(getattr(value, name))
        )
    }

