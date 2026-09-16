"""Read YAML settings; Python remains responsible for calculations and behavior."""

from pathlib import Path
from collections.abc import Mapping

import numpy as np
import yaml

CONFIG_DIR = Path(__file__).resolve().parent / "config"


class Settings(dict):
    """A settings mapping with attribute access for simulation code."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name, value):
        self[name] = value


def settings(value):
    if isinstance(value, dict):
        return Settings({key: settings(item) for key, item in value.items()})
    if isinstance(value, list):
        return [settings(item) for item in value]
    return value


class SettingsLoader(yaml.SafeLoader):
    """Safe YAML plus an explicit NumPy-array tag for controller matrices."""


def _array(loader, node):
    return np.asarray(loader.construct_sequence(node, deep=True))


def _diagonal(loader, node):
    return np.diag(loader.construct_sequence(node, deep=True))


SettingsLoader.add_constructor("!array", _array)
SettingsLoader.add_constructor("!diag", _diagonal)


def read_yaml(path):
    """Load a mapping from a user file or the packaged config directory."""
    path = Path(path)
    if not path.is_file():
        path = CONFIG_DIR / path
    with path.open() as stream:
        values = yaml.load(stream, Loader=SettingsLoader)
    if not isinstance(values, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return values


def load_settings(path):
    return settings(read_yaml(path))


def flatten_overrides(values, prefix=""):
    """Normalize nested YAML and dotted Python overrides to the same paths."""
    result = {}
    for name, value in values.items():
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(value, Mapping):
            entries = flatten_overrides(value, path)
        else:
            entries = {path: value}
        for key, item in entries.items():
            if key in result:
                raise ValueError(f"Duplicate override path: {key}")
            result[key] = item
    return result


def rl_overrides(values):
    """Select RL settings from controller overrides; bare RL mappings remain valid."""
    flat = flatten_overrides(values or {})
    if any(path.startswith("RL.") for path in flat):
        if not all(path.startswith("RL.") for path in flat):
            raise ValueError(
                "An RL override file cannot mix RL paths with other sections"
            )
        return {path[3:]: value for path, value in flat.items()}
    return flat


def apply_overrides(config, values):
    """Apply known settings only; reject typos before constructing derived state."""
    for path, value in flatten_overrides(values).items():
        target = config
        parts = path.split(".")
        for part in parts[:-1]:
            if not hasattr(target, part):
                raise ValueError(f"Unknown override path: {path}")
            target = getattr(target, part)
        if not hasattr(target, parts[-1]):
            raise ValueError(f"Unknown override path: {path}")
        setattr(target, parts[-1], value)


def resolve_algorithm(algorithm, overrides, default):
    """An explicit algorithm and an override must agree; neither silently wins."""
    requested = overrides.get("algorithm")
    if algorithm is not None and requested is not None:
        if algorithm.lower() != requested.lower():
            raise ValueError(
                "Conflicting algorithm settings in arguments and overrides"
            )
    selected = algorithm if algorithm is not None else requested
    selected = default if selected is None else selected.lower()
    if selected not in ("ppo", "vpg", "sac"):
        raise ValueError("algorithm must be ppo, vpg or sac")
    return selected
