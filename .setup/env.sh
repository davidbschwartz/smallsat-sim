# Source from Bash or Zsh, from any working directory.
# The wrapper also sources this before launching uv.
if [ -n "${BASH_VERSION:-}" ]; then
    _smallsat_env_file=${BASH_SOURCE[0]}
elif [ -n "${ZSH_VERSION:-}" ]; then
    _smallsat_env_file=${(%):-%x}
else
    echo "Source .setup/env.sh from Bash or Zsh." >&2
    return 1
fi
_smallsat_root=$(cd "$(dirname "$_smallsat_env_file")/.." && pwd)
export ACADOS_SOURCE_DIR="$_smallsat_root/deps/acados"
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [ "$(uname -s)" = Darwin ]; then
    export DYLD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi
# EGL supports headless rendering on the primary Linux/NVIDIA target.
if [ "$(uname -s)" = Linux ]; then
    export MUJOCO_GL=${MUJOCO_GL:-egl}
fi
unset _smallsat_env_file _smallsat_root
