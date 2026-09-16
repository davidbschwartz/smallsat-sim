"""Check dependency imports and shared libraries."""
import argparse
import ctypes
import importlib
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mpc", action="store_true")
parser.add_argument("--gpu", action="store_true")
args = parser.parse_args()

for name in ("mujoco", "mujoco.mjx", "jax", "viser"):
    importlib.import_module(name)
    print(f"OK: {name}")

if args.mpc:
    for name in ("acados_template", "l4acados", "torch", "gpytorch", "linear_operator"):
        importlib.import_module(name)
        print(f"OK: {name}")
    root = Path(os.environ["ACADOS_SOURCE_DIR"])
    suffix = "dylib" if sys.platform == "darwin" else "so"
    for name in (f"libblasfeo.{suffix}", f"libhpipm.{suffix}", f"libacados.{suffix}"):
        ctypes.CDLL(str(root / "lib" / name), mode=ctypes.RTLD_GLOBAL)
        print(f"OK: {name}")
    renderer = root / "bin" / "t_renderer"
    if not renderer.is_file() or not os.access(renderer, os.X_OK):
        raise RuntimeError(f"Missing executable: {renderer}")
    print(f"OK: {renderer}")

if args.gpu:
    import jax
    import warp

    print("JAX devices:", jax.devices())
    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("JAX cannot see a CUDA GPU")
    warp.init()
    if not warp.is_cuda_available():
        raise RuntimeError("Warp cannot see a CUDA GPU")
    print("OK: CUDA is visible to JAX and Warp")
