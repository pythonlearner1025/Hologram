#!/usr/bin/env bash
#------------------------------------------------------------------------------
# Installs GPU-enabled JAX (CUDA 12 wheel) + JWAVE in an isolated venv.
#
# 1. Creates ~/jaxenv        – keep all JAX stuff out of the system site-pkgs
# 2. Upgrades pip & build tools
# 3. Pulls the last pre-0.5 CUDA-12 build of JAX (0.4.38)  ➜  satisfies jwave
# 4. Installs jwave
# 5. Prints the detected JAX devices for a sanity check
#
# Tested on:
#   • runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04
#   • NVIDIA RTX 4090, driver ≥ 550
#------------------------------------------------------------------------------

set -euo pipefail

JAX_VERSION="0.4.38"          # <0.5.0 keeps jaxdf/jwave happy
VENV_HOME="/workspace/hologram/jaxenv"    # change if you like

echo ">>> Creating virtual-env at ${VENV_HOME}"
python3 -m venv "${VENV_HOME}"
# shellcheck source=/dev/null
source "${VENV_HOME}/bin/activate"

echo ">>> Upgrading pip, setuptools, wheel"
pip install -U pip setuptools wheel

echo ">>> Installing JAX ${JAX_VERSION} with CUDA 12 support"
pip install -U \
  "jax[cuda12]==${JAX_VERSION}" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

echo ">>> Installing JWAVE"
pip install jwave

echo ">>> Verifying GPU is visible to JAX"
python - <<'PY'
import jax, os, sys, subprocess
print("JAX          :", jax.__version__)
print("CUDA devices :", jax.devices())
# Extra: show nvidia-smi one-liner if available
try:
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    print("nvidia-smi   :", out.decode().strip())
except Exception:
    pass
PY

echo ">>> Done – activate later via: source ${VENV_HOME}/bin/activate"
