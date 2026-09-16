"""Vectorized simulation. Import functional kernels without loading runtime viewers."""


def __getattr__(name):
    if name == "VecEnv":
        from .runtime import VecEnv
        return VecEnv
    raise AttributeError(name)


__all__ = ["VecEnv"]
