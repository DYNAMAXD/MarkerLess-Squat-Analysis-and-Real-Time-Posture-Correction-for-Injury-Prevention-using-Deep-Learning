# start_batch.ps1 -- one-click launcher for the batch preprocessing page (whole dataset -> CSV files).
#
#   Right-click > Run with PowerShell   (or:  powershell -ExecutionPolicy Bypass -File two_view\start_batch.ps1)
#
# First run (needs internet, ~10-15 min): installs Python 3.11 if missing (winget), creates C:\Pose2SimVenv with the
# exact package versions of two_view\requirements_pose2sim.txt, and - when an NVIDIA GPU is present - swaps the CPU
# onnxruntime for onnxruntime-gpu with its CUDA/cuDNN libraries (only the NVIDIA driver is needed, no CUDA toolkit).
# The pose models (~300 MB) are downloaded by the first person processed.
# Then it opens http://127.0.0.1:5050 : give the dataset folder, check the settings, press Start.
# KEEP THIS WINDOW OPEN while the batch runs (closing it stops the batch; starting again resumes).
#
# Without the page:  .\two_view\start_batch.ps1 -Dataset D:\squat_dataset -ExtraArgs "--intr-square 41 --ext-square 41"

param(
    [string]$Dataset = "",     # run the batch directly on this folder (no web page)
    [string]$ExtraArgs = "",   # more batch_preprocess.py options for -Dataset
    [switch]$ForceCpu          # install the CPU onnxruntime even on a GPU machine
)
# "Continue": in Windows PowerShell 5.1 "Stop" turns any stderr line of python/pip into a fatal error; failures are
# checked through $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"
$venv = "C:\Pose2SimVenv"      # outside the repo: its long folder name can hit the Windows path-length limit
$py   = "$venv\Scripts\python.exe"
$repo = Split-Path -Parent $PSScriptRoot
$req  = "$repo\two_view\requirements_pose2sim.txt"

function Need-Python311 {
    try { & py -3.11 -c "import sys" 2>$null; if ($LASTEXITCODE -eq 0) { return } } catch {}
    Write-Host "Python 3.11 not found; installing it with winget ..." -ForegroundColor Yellow
    winget install --id Python.Python.3.11 -e --accept-source-agreements --accept-package-agreements
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
    & py -3.11 -c "import sys"
    if ($LASTEXITCODE -ne 0) { throw "Python 3.11 is still not available. Install it from python.org (tick 'py launcher') and run this again." }
}

if (-not (Test-Path $py)) {
    Need-Python311
    Write-Host "First run: creating the Pose2Sim environment at $venv ..." -ForegroundColor Yellow
    py -3.11 -m venv $venv
    & $py -m pip install --upgrade pip
    & $py -m pip install -r $req
    if ($LASTEXITCODE -ne 0) { throw "Installing the packages failed (see above)." }
}

# GPU: onnxruntime-gpu instead of onnxruntime. Both install the same 'onnxruntime' module, so remove both first.
$hasGpu = $null -ne (Get-Command nvidia-smi -ErrorAction SilentlyContinue)
if ($hasGpu -and -not $ForceCpu) {
    & $py -c "import onnxruntime as o, sys; o.preload_dlls() if hasattr(o, 'preload_dlls') else None; sys.exit(0 if 'CUDAExecutionProvider' in o.get_available_providers() else 1)" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "NVIDIA GPU found: installing onnxruntime-gpu with CUDA + cuDNN ..." -ForegroundColor Yellow
        & $py -m pip uninstall -y onnxruntime onnxruntime-gpu
        & $py -m pip install "onnxruntime-gpu[cuda,cudnn]==1.30.0"
        if ($LASTEXITCODE -ne 0) { & $py -m pip install "onnxruntime-gpu[cuda,cudnn]" }
    }
} else {
    & $py -c "import onnxruntime" 2>$null
    if ($LASTEXITCODE -ne 0) { & $py -m pip install "onnxruntime==1.30.0" }
}
& $py -c "import onnxruntime as o; o.preload_dlls() if hasattr(o, 'preload_dlls') else None; p = o.get_available_providers(); print('onnxruntime', o.__version__, '->', 'GPU (CUDA)' if 'CUDAExecutionProvider' in p else 'CPU only', p)"
if ($hasGpu) {
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    # onnxruntime-gpu 1.30 ships CUDA 13 libraries, which need driver 580+. Older drivers fail with CUDA error 801/35
    # and the batch silently falls back to the (much slower) CPU.
    $drv = (nvidia-smi --query-gpu=driver_version --format=csv,noheader | Select-Object -First 1).Trim()
    if ([int]($drv.Split('.')[0]) -lt 580) {
        Write-Host "WARNING: NVIDIA driver $drv is too old for the GPU build (CUDA 13 needs driver 580 or newer)." -ForegroundColor Red
        Write-Host "         Update it (nvidia.com/drivers or the NVIDIA app), then run this again. Otherwise everything runs on the CPU." -ForegroundColor Red
    }
}

Set-Location $repo
if ($Dataset) {
    Write-Host "Running the batch on $Dataset (Ctrl+C to stop; running it again resumes) ..." -ForegroundColor Green
    $cmd = "& `"$py`" -u `"$repo\two_view\batch_preprocess.py`" --input `"$Dataset`" $ExtraArgs"
    Invoke-Expression $cmd
    exit $LASTEXITCODE
}
Write-Host "Starting the batch page at http://127.0.0.1:5050  (keep this window open while the batch runs)" -ForegroundColor Green
Start-Job { Start-Sleep -Seconds 4; Start-Process "http://127.0.0.1:5050" } | Out-Null
& $py "$repo\two_view\batch_app.py"
