#!/usr/bin/env bash
#------------------------------------------------------------------------------
# Installs CPU-only JAX + JWAVE in an isolated venv.
#
# 1. Creates ~/cpuenv        – keep all JAX stuff out of the system site-pkgs
# 2. Upgrades pip & build tools
# 3. Pulls the last pre-0.5 CPU build of JAX (0.4.38)  ➜  satisfies jwave
# 4. Installs jwave
# 5. Prints the detected JAX devices for a sanity check
#
# Tested on:
#   • Various CPU environments
#   • macOS and Linux systems
#------------------------------------------------------------------------------

set -euo pipefail

JAX_VERSION="0.4.38"          # <0.5.0 keeps jaxdf/jwave happy
VENV_HOME="$(pwd)/cpuenv"    # create venv in current directory

echo ">>> Creating virtual-env at ${VENV_HOME}"
python3 -m venv "${VENV_HOME}"
# shellcheck source=/dev/null
source "${VENV_HOME}/bin/activate"

echo ">>> Upgrading pip, setuptools, wheel"
pip install -U pip setuptools wheel

echo ">>> Installing JAX ${JAX_VERSION} with CPU support"
pip install -U "jax[cpu]==${JAX_VERSION}"

echo ">>> Installing JWAVE and dependencies"
pip install jwave optax matplotlib

echo ">>> Verifying CPU is visible to JAX"
python - <<'PY'
import jax, os, sys
print("JAX          :", jax.__version__)
print("CPU devices  :", jax.devices())
print("Platform     :", jax.default_backend())
PY

echo ">>> Done – activate later via: source ${VENV_HOME}/bin/activate"