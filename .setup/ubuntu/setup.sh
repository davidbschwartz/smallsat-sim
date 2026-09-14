#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
gpu=1
mpc=1
system_deps=0
for arg in "$@"; do
    case "$arg" in
        --cpu) gpu=0 ;;
        --without-mpc) mpc=0 ;;
        --system-deps) system_deps=1 ;;
        -h|--help)
            echo "Usage: bash .setup/ubuntu/setup.sh [--cpu] [--without-mpc] [--system-deps]"
            echo "Defaults: CUDA 12, Warp, and the full native MPC dependency stack."
            echo "--system-deps installs Ubuntu packages using sudo apt-get."
            exit 0 ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done
[[ $(uname -s) == Linux ]] || { echo "This installer targets Linux (Ubuntu 22.04+)." >&2; exit 1; }
command -v uv >/dev/null || { echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/" >&2; exit 1; }
if (( gpu )); then
    command -v nvidia-smi >/dev/null || { echo "Install an NVIDIA driver first, or use --cpu." >&2; exit 1; }
    nvidia-smi
fi
if (( system_deps )); then
    sudo apt-get update
    packages=(git build-essential cmake pkg-config libopenblas-dev liblapack-dev
        libegl1 libgles2 libgl1 libglfw3 libglew-dev libosmesa6 libosmesa6-dev
        libsm6 libxext6 libgomp1 ffmpeg)
    if (( mpc )); then packages+=(cargo rustc); fi
    sudo apt-get install -y "${packages[@]}"
fi
cd "$repo_root"
if (( mpc )); then bash .setup/native/install_mpc.sh; fi
sync_args=(sync --locked)
checks=(python .setup/check_install.py)
if (( gpu )); then sync_args+=(--extra cuda12 --extra warp); checks+=(--gpu); fi
if (( mpc )); then sync_args+=(--extra mpc); checks+=(--mpc); fi
uv "${sync_args[@]}"
source .setup/env.sh
uv run --no-sync "${checks[@]}"
printf '\nSetup complete. Run: bash .setup/smallsat run python experiments/test.py --headless\n'
