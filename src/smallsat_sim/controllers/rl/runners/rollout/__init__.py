"""A scan collects transitions; the collector supplies policy and context behavior."""

from .core import run_functional_rollout
from .types import FunctionalRolloutCallbacks, FunctionalRolloutResult
