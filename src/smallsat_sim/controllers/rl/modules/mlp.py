"""Small seeded MLPs shared by policy and value models."""

import math

from flax import nnx


def mlp(
    sizes,
    activation=nnx.tanh,
    output_activation=None,
    std=math.sqrt(2),
    last_layer_std=1.0,
    *,
    rngs,
):
    layers = []
    for i, (inputs, outputs) in enumerate(zip(sizes[:-1], sizes[1:])):
        last = i == len(sizes) - 2
        layers.append(
            nnx.Linear(
                inputs,
                outputs,
                rngs=rngs,
                kernel_init=nnx.initializers.orthogonal(
                    last_layer_std if last else std
                ),
                bias_init=nnx.initializers.zeros_init(),
            )
        )
        fn = output_activation if last else activation
        if fn is not None:
            layers.append(fn)
    return nnx.Sequential(*layers)
