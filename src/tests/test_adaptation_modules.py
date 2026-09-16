"""Adaptation-module shape, history and update contract tests."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from smallsat_sim.controllers.rl.modules.am_cnn import CNNAdaptationModule
from smallsat_sim.controllers.rl.modules.am_transformer import TransformerAdaptationModule
from smallsat_sim.controllers.rl.runners.rollout.context import ContextHistory, append_history, reset_history, estimate_context
from smallsat_sim.controllers.rl.runners.adaptation_training import gather_windows, adaptation_updates
import optax


@pytest.mark.parametrize("cls", [CNNAdaptationModule, TransformerAdaptationModule])
@pytest.mark.parametrize("steps", [5, 8, 13])
def test_batched_history_shapes_and_gradients(cls, steps):
    model = cls(steps, 4, 6, rngs=nnx.Rngs(0))
    history = jax.random.normal(jax.random.PRNGKey(1), (2, steps, 4))
    prediction = model(history)
    assert prediction.shape == (2, 6)
    np.testing.assert_allclose(prediction[0], model(history[0]), atol=1e-6)
    gradients = nnx.grad(lambda m: jnp.square(m(history)).mean())(model)
    assert all(np.all(np.isfinite(x)) for x in jax.tree.leaves(gradients))
    with pytest.raises(ValueError, match="history"):
        model(jnp.zeros((steps + 1, 4)))


def test_history_alignment_and_independent_reset():
    history = ContextHistory(jnp.zeros((2, 3, 2)), jnp.zeros(2, dtype=jnp.int32))
    features = []
    for t in range(3):
        states, actions = jnp.full((2, 1), t), jnp.full((2, 1), t + 10)
        features.append(jnp.concatenate((states, actions), axis=-1))
        history = append_history(history, states, actions)
    np.testing.assert_array_equal(history.values, [[[0, 10], [1, 11], [2, 12]]] * 2)
    np.testing.assert_array_equal(history.counts, [3, 3])
    features = jnp.stack(features)
    # c_(t+1) has a distinct label from c_t; windows end at the label transition.
    targets = jnp.broadcast_to(jnp.arange(1, 4)[:, None, None], (3, 2, 1))
    x, y = gather_windows(features, targets, jnp.array([[2, 0]]), 3)
    np.testing.assert_array_equal(x[0], history.values[0])
    assert y[0, 0] == 3
    cleared = reset_history(history, jnp.array([True, False]))
    assert not cleared.values[0].any()
    np.testing.assert_array_equal(cleared.counts, [0, 3])
    np.testing.assert_array_equal(cleared.values[1], history.values[1])
    model = lambda x: jnp.ones((x.shape[0], 6))
    context = estimate_context(model, cleared, jnp.ones(6))
    np.testing.assert_array_equal(context[0], jnp.zeros(6))
    np.testing.assert_array_equal(context[1], jnp.ones(6))


def test_adaptation_minibatches_fixed_shape_and_environment_split():
    model = CNNAdaptationModule(3, 2, 6, rngs=nnx.Rngs(0))
    opt = nnx.Optimizer(model, optax.adam(0.01))
    graph, state = nnx.split((model, opt))
    features = jnp.ones((5, 4, 2))
    targets = jnp.zeros((5, 4, 6))
    valid = jnp.arange(5)[:, None] >= jnp.full((1, 4), 2)
    _, metrics = adaptation_updates(graph, state, features, targets, valid,
        jax.random.PRNGKey(1), history_len=3, batch_size=4, updates=2)
    assert metrics["train_windows"] == 9
    assert metrics["validation_windows"] == 3
    assert jnp.isfinite(metrics["am_val_loss_mean"])
