#!/usr/bin/env bash
# Wrapper to initialize Isaac Sim environment and then exec the project's conda python
# This script will be used as the "python" interpreter in VS Code launch.json

# Path to Isaac Sim installation
ISAAC_ROOT="/opt/isaac/isaacsim-5.1"
ISAAC_SETUP="$ISAAC_ROOT/setup_conda_env.sh"
PYTHON_BIN="/home/yzy/.conda/envs/isaaclab230/bin/python"

# Source Isaac Sim env setup if available
if [ -f "$ISAAC_SETUP" ]; then
    # shellcheck disable=SC1090
    source "$ISAAC_SETUP"
fi

# Ensure necessary library and app paths are set
export LD_LIBRARY_PATH="$ISAAC_ROOT/kit/lib:${LD_LIBRARY_PATH}"
export PATH="$ISAAC_ROOT/kit/bin:${PATH}"
export EXP_PATH="$ISAAC_ROOT/apps"
export PYTHONPATH="$ISAAC_ROOT/kit/python/lib/python3.11/site-packages:$ISAAC_ROOT/python_packages:$ISAAC_ROOT/exts/isaacsim.simulation_app:${PYTHONPATH}"

# Exec the real python interpreter from the conda env with all passed arguments
exec "$PYTHON_BIN" "$@"
