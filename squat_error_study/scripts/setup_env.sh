#!/usr/bin/env bash
# =============================================================================
# setup_env.sh  -  create the Python environment for the squat study (Linux / macOS)
# =============================================================================
# Usage (from the squat_error_study folder):
#   bash scripts/setup_env.sh            # CPU-only PyTorch, venv at ~/squat_venv
#   bash scripts/setup_env.sh gpu        # NVIDIA GPU build of PyTorch (lab machine)
#   VENV=/data/envs/squat PYTHON=python3.11 bash scripts/setup_env.sh gpu
set -euo pipefail
VENV="${VENV:-$HOME/squat_venv}"
PYTHON="${PYTHON:-python3}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "1/3  creating venv at $VENV"
"$PYTHON" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip

echo "2/3  installing requirements"
if [ "${1:-}" = "gpu" ]; then
  "$VENV/bin/python" -m pip install -r "$ROOT/requirements.txt" --extra-index-url https://download.pytorch.org/whl/cu121
else
  "$VENV/bin/python" -m pip install -r "$ROOT/requirements.txt" --extra-index-url https://download.pytorch.org/whl/cpu
fi

echo "3/3  check"
"$VENV/bin/python" -c "import torch,sklearn,xgboost,pandas;print('torch',torch.__version__,'cuda available:',torch.cuda.is_available())"
echo "Done. Use it with:  $VENV/bin/python scripts/run_cv.py"
