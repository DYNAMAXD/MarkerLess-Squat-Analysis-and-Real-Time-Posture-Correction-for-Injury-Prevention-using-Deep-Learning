# =============================================================================
# setup_env.ps1  -  create the Python environment for the squat study (Windows)
# =============================================================================
# Usage (PowerShell, from the squat_error_study folder):
#   .\scripts\setup_env.ps1                 # CPU-only PyTorch, venv at C:\squat_venv
#   .\scripts\setup_env.ps1 -Gpu            # NVIDIA GPU build of PyTorch (lab machine)
#   .\scripts\setup_env.ps1 -Venv D:\envs\squat -Python 3.11
#
# Why the venv is NOT inside the repo: the repo folder name is very long and Windows' 260-character
# path limit breaks PyTorch's install. A short path such as C:\squat_venv avoids that.
param(
    [string]$Venv = "C:\squat_venv",
    [string]$Python = "3.11",
    [switch]$Gpu
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

Write-Host "1/3  creating venv at $Venv (python $Python)"
py -$Python -m venv $Venv
& "$Venv\Scripts\python.exe" -m pip install --upgrade pip

Write-Host "2/3  installing requirements"
if ($Gpu) {
    # CUDA 12.1 wheels; change cu121 if the lab driver needs another build (see pytorch.org)
    & "$Venv\Scripts\python.exe" -m pip install -r "$root\requirements.txt" --extra-index-url https://download.pytorch.org/whl/cu121
} else {
    & "$Venv\Scripts\python.exe" -m pip install -r "$root\requirements.txt" --extra-index-url https://download.pytorch.org/whl/cpu
}

Write-Host "3/3  check"
& "$Venv\Scripts\python.exe" -c "import torch,sklearn,xgboost,pandas;print('torch',torch.__version__,'cuda available:',torch.cuda.is_available())"
Write-Host "Done. Use it with:  $Venv\Scripts\python.exe scripts\run_cv.py"
