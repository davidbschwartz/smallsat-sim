"""Checkpoint names identify runs; metadata validates their exact configuration."""

from pathlib import Path
import os

from smallsat_sim.api.runs import build_checkpoint_file_names as run_names, RunSpec


def configure_jax_compilation_cache(jax_module):
    # Cache eligible JAX compilations on disk for reuse across runs.
    if os.environ.get("SMALLSAT_DISABLE_JAX_CACHE") == "1":
        return
    path = Path(os.environ.get("SMALLSAT_JAX_CACHE_DIR", ".cache/jax"))
    path.mkdir(parents=True, exist_ok=True)
    jax_module.config.update("jax_compilation_cache_dir", str(path.resolve()))


def build_checkpoint_file_names(env, algorithm):
    spec = getattr(env, "run_spec", None)
    if spec is None:
        spec = RunSpec(
            vehicle=env.env_cfg.model,
            name=f"{env.run_name}_{algorithm}_seed{env.env_cfg.sim.seed}",
        )
    return run_names(spec)
