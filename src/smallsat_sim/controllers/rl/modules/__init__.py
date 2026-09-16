"""Policy, value and history-estimation networks used by the retained RL methods."""

from flax import nnx

from .am_cnn import CNNAdaptationModule
from .am_transformer import TransformerAdaptationModule


def adaptation_model(architecture, *, history_length, input_size, context_size, key):
    """Select an estimator explicitly; None means a policy without adaptation."""
    if architecture is None:
        return None

    rngs = nnx.Rngs(key)
    if architecture == "cnn":
        return CNNAdaptationModule(history_length, input_size, context_size, rngs=rngs)
    if architecture == "transformer":
        return TransformerAdaptationModule(
            history_length, input_size, context_size, rngs=rngs
        )
    raise ValueError(f"Unknown adaptation architecture: {architecture}")
