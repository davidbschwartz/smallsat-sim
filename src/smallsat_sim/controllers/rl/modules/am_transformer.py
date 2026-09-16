"""Small deterministic temporal transformer for the CNN comparison."""

from flax import nnx
import jax.numpy as jnp


class AttentionBlock(nnx.Module):
    def __init__(self, width, heads, *, rngs):
        self.norm1 = nnx.LayerNorm(width, rngs=rngs)
        self.attention = nnx.MultiHeadAttention(heads, width, decode=False, rngs=rngs)
        self.norm2 = nnx.LayerNorm(width, rngs=rngs)
        self.up = nnx.Linear(width, 2 * width, rngs=rngs)
        self.down = nnx.Linear(2 * width, width, rngs=rngs)

    def __call__(self, x):
        x = x + self.attention(self.norm1(x), decode=False)
        return x + self.down(nnx.gelu(self.up(self.norm2(x))))


class TransformerAdaptationModule(nnx.Module):
    def __init__(
        self,
        n_steps,
        state_action_dim,
        ext_dim,
        *,
        rngs,
        d_model=64,
        n_heads=4,
        n_layers=2,
    ):
        if min(n_steps, state_action_dim, ext_dim, n_layers) <= 0 or d_model % n_heads:
            raise ValueError(
                "Positive dimensions and width divisible by heads required"
            )
        self.n_steps, self.state_action_dim = n_steps, state_action_dim
        self.encoder = nnx.Linear(state_action_dim, d_model, rngs=rngs)
        positions = jnp.arange(n_steps)[:, None]
        frequency = jnp.exp(-jnp.log(10000.0) * jnp.arange(0, d_model, 2) / d_model)
        self.positions = jnp.stack(
            (jnp.sin(positions * frequency), jnp.cos(positions * frequency)), axis=-1
        ).reshape(n_steps, d_model)
        self.blocks = [
            AttentionBlock(d_model, n_heads, rngs=rngs) for _ in range(n_layers)
        ]
        self.norm = nnx.LayerNorm(d_model, rngs=rngs)
        self.output = nnx.Linear(d_model, ext_dim, rngs=rngs)

    def __call__(self, history):
        if history.shape[-2:] != (self.n_steps, self.state_action_dim):
            raise ValueError("Expected (..., history_length, state_action_dim) history")
        # All tokens belong to the completed history; no future observations exist.
        x = self.encoder(history) + self.positions
        for block in self.blocks:
            x = block(x)
        return self.output(self.norm(x)[..., -1, :])
