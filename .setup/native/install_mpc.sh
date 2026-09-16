#!/usr/bin/env bash
# Build native libraries only. Python dependencies are managed by uv's mpc extra.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
case "$(uname -s)" in
    Linux|Darwin) ;;
    *) echo "The native MPC build supports Linux and macOS." >&2; exit 1 ;;
esac
for command in git cmake make cc cargo rustc; do
    command -v "$command" >/dev/null || { echo "Missing build tool: $command" >&2; exit 1; }
done
rust_version=$(rustc --version | awk '{print $2}')
IFS=. read -r rust_major rust_minor _ <<< "$rust_version"
if (( rust_major < 1 || (rust_major == 1 && rust_minor < 75) )); then
    echo "Rust 1.75+ is required. Update Rust before building the MPC renderer." >&2
    exit 1
fi
acados_dir="$repo_root/deps/acados"
# Use the very same revision as the Python interface, with no second version pin.
acados_rev=$(sed -n 's/^acados-template = .*rev = "\([a-f0-9]*\)".*/\1/p' "$repo_root/pyproject.toml")
[[ ${#acados_rev} == 40 ]] || { echo "Cannot read the acados revision from pyproject.toml" >&2; exit 1; }
if [[ ! -e "$acados_dir" ]]; then
    mkdir -p "$repo_root/deps"
    git clone --no-checkout https://github.com/acados/acados.git "$acados_dir"
    git -C "$acados_dir" checkout --detach "$acados_rev"
fi
if [[ $(git -C "$acados_dir" rev-parse HEAD) != "$acados_rev" ]]; then
    echo "Existing $acados_dir has a different revision. Move it aside and rerun setup." >&2
    exit 1
fi
if [[ -n $(git -C "$acados_dir" status --porcelain --untracked-files=no) ]]; then
    echo "Existing acados checkout has changes; refusing to build a modified dependency." >&2
    exit 1
fi
git -C "$acados_dir" submodule update --init --recursive
cmake -S "$acados_dir" -B "$acados_dir/build" \
    -DCMAKE_INSTALL_PREFIX="$acados_dir" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DACADOS_WITH_QPOASES=ON -DACADOS_SILENT=ON \
    -DBLASFEO_TARGET=GENERIC -DHPIPM_TARGET=GENERIC
cmake --build "$acados_dir/build" --parallel "${SMALLSAT_BUILD_JOBS:-2}"
cmake --install "$acados_dir/build"
(
    cd "$acados_dir/interfaces/acados_template/tera_renderer"
    # Upstream does not ship a lockfile; keep transitive Rust dependencies fixed.
    cp "$repo_root/.setup/native/tera-Cargo.lock" Cargo.lock
    cargo build --release --locked
    mkdir -p "$acados_dir/bin"
    install -m 755 target/release/t_renderer "$acados_dir/bin/t_renderer"
)
echo "Native MPC libraries installed in $acados_dir"
