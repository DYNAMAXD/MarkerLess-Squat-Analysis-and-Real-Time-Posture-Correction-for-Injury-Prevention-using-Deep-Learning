# setup.ps1 -- prepares a fresh clone for the final product. final_product.ps1 runs it on every start; when everything
# is already there it only checks (a few seconds). It can also be run alone:
#
#   powershell -ExecutionPolicy Bypass -File final_product\setup.ps1            (-ForceCpu: no GPU packages)
#
# First run (needs internet, ~15-25 min, ~6 GB next to the clone):
#   1. the large files kept in Git LFS (MotionBERT weights, SMPL body model) - pulled if the clone only has pointers
#   2. Windows long paths switched on (one administrator prompt) if they are off: the environments live inside the
#      clone and their deepest files (torch) are ~190 characters below the environment folder
#   3. Python 3.11 (winget) if missing
#   4. three environments inside the clone, in .venv\ (git-ignored):
#        .venv\app        app + 3D body mesh             (final_product\requirements_app.txt)
#        .venv\pose2sim   sync, calibration, 2D pose, 3D (two_view\requirements_pose2sim.txt; onnxruntime-gpu with
#                         its own CUDA libraries when an NVIDIA GPU with driver >= 580 is present, else CPU)
#        .venv\model      error-detection model           (final_product\requirements_model.txt + CPU torch)
#      An environment that already exists is kept and only completed if a package is missing.
# The pose models (~300 MB) are downloaded by the first analysis.

param(
    [string]$AppVenv = (Join-Path (Split-Path -Parent $PSScriptRoot) ".venv\app"),
    [string]$Pose2SimVenv = (Join-Path (Split-Path -Parent $PSScriptRoot) ".venv\pose2sim"),
    [string]$ModelVenv = (Join-Path (Split-Path -Parent $PSScriptRoot) ".venv\model"),
    [switch]$ForceCpu
)
# "Continue": in Windows PowerShell 5.1 "Stop" turns any stderr line of python/pip/git into a fatal error; failures
# are checked through $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
$failed = @()

function Say($text, $colour = "Yellow") { Write-Host $text -ForegroundColor $colour }

# --- 0. Windows long paths ------------------------------------------------------------------------------------------
# Windows refuses paths over 260 characters unless LongPathsEnabled is on; pip then fails half-way through torch.
$deepest = 190   # longest file path inside an environment (torch's bundled licence folders), measured 2026-10-19
$longest = (@($AppVenv, $Pose2SimVenv, $ModelVenv) | ForEach-Object { $_.Length } | Measure-Object -Maximum).Maximum
function Test-LongPaths {
    try { return (Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -ErrorAction Stop).LongPathsEnabled -eq 1 }
    catch { return $false }
}
if ($longest + $deepest -ge 260 -and -not (Test-LongPaths)) {
    Say "This folder's path is long and Windows long paths are off, so the packages cannot be installed here yet."
    Say "Turning on Windows long paths (one time) - please click Yes in the administrator prompt ..."
    try {
        Start-Process powershell -Verb RunAs -Wait -ArgumentList '-NoProfile -Command "Set-ItemProperty -Path HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem -Name LongPathsEnabled -Value 1 -Type DWord"'
    } catch {}
    if (-not (Test-LongPaths)) {
        Say "Windows long paths are still off. Either start again and click Yes in the administrator prompt, or clone the repository into a short folder (for example C:\squat) and start it from there." "Red"
        exit 1
    }
    Say "Windows long paths are on." "Green"
}

# --- 1. Git LFS files ---------------------------------------------------------------------------------------------
# Without git-lfs a clone holds ~130-byte text pointers instead of the weights.
$lfsFiles = @(
    "MotionBERT\checkpoint\mesh\FT_MB_release_MB_ft_pw3d\best_epoch.bin",
    "MotionBERT\data\mesh\SMPL_NEUTRAL.pkl",
    "MotionBERT\data\mesh\J_regressor_extra.npy",
    "MotionBERT\data\mesh\J_regressor_h36m_correct.npy"
)
function Test-LfsPointer($path) {
    if (-not (Test-Path $path)) { return $true }
    if ((Get-Item $path).Length -gt 1024) { return $false }
    return (Get-Content $path -TotalCount 1) -like "version https://git-lfs*"
}
$missingLfs = @($lfsFiles | Where-Object { Test-LfsPointer (Join-Path $repo $_) })
if ($missingLfs.Count -gt 0) {
    Say "The large model files (Git LFS) are not downloaded yet: $($missingLfs -join ', ')"
    if (-not (Get-Command git -ErrorAction SilentlyContinue) -or -not (Test-Path (Join-Path $repo ".git"))) {
        Say "This folder is not a git clone. Clone the repository with git (git-lfs installed) instead of downloading a ZIP." "Red"
        $failed += "Git LFS files"
    } else {
        & git lfs version 2>$null | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Say "Installing Git LFS with winget ..."
            winget install --id GitHub.GitLFS -e --accept-source-agreements --accept-package-agreements
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
        }
        Say "Downloading them (git lfs pull, ~400 MB) ..."
        & git -C $repo lfs install --local
        & git -C $repo lfs pull
        $missingLfs = @($lfsFiles | Where-Object { Test-LfsPointer (Join-Path $repo $_) })
        if ($missingLfs.Count -gt 0) {
            Say "Git LFS download failed. Install Git LFS (https://git-lfs.com), then run 'git lfs pull' in $repo ." "Red"
            $failed += "Git LFS files"
        }
    }
}

# --- 2. Python 3.11 -------------------------------------------------------------------------------------------------
function Test-Python311 {
    try { & py -3.11 -c "import sys" 2>$null; return $LASTEXITCODE -eq 0 } catch { return $false }
}
function Need-Python311 {
    if (Test-Python311) { return $true }
    Say "Python 3.11 not found; installing it with winget ..."
    winget install --id Python.Python.3.11 -e --accept-source-agreements --accept-package-agreements
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
    if (Test-Python311) { return $true }
    Say "Python 3.11 is still not available. Install it from python.org (tick 'py launcher'), then start again." "Red"
    return $false
}

# Creates the environment if missing; returns its python.exe, or $null when it cannot be created.
function New-Env($venv) {
    $py = Join-Path $venv "Scripts\python.exe"
    if (Test-Path $py) { return $py }
    if (-not (Need-Python311)) { return $null }
    Say "Creating the environment $venv ..."
    & py -3.11 -m venv $venv
    if (-not (Test-Path $py)) { Say "Could not create $venv." "Red"; return $null }
    & $py -m pip install --upgrade pip --quiet
    return $py
}

function Test-Imports($py, $modules) {
    & $py -c "import $modules" 2>$null
    return $LASTEXITCODE -eq 0
}

# GPU builds need an NVIDIA driver new enough for their CUDA version; returns the driver's major version or 0.
$driverMajor = 0
if (-not $ForceCpu -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    try { $driverMajor = [int]((nvidia-smi --query-gpu=driver_version --format=csv,noheader | Select-Object -First 1).Trim().Split('.')[0]) } catch {}
}

# --- 3a. App environment --------------------------------------------------------------------------------------------
$appPy = New-Env $AppVenv
if ($appPy -and -not (Test-Imports $appPy "flask, numpy, cv2, scipy, pandas, torch, smplx, easydict, yaml, imageio_ffmpeg")) {
    Say "Installing the app packages into $AppVenv (torch, smplx, flask, ...) ..."
    & $appPy -m pip install -r (Join-Path $PSScriptRoot "requirements_app.txt")
    if (-not (Test-Imports $appPy "flask, numpy, cv2, scipy, pandas, torch, smplx, easydict, yaml, imageio_ffmpeg")) { $failed += $AppVenv }
} elseif (-not $appPy) { $failed += $AppVenv }

# --- 3b. Pose2Sim environment ---------------------------------------------------------------------------------------
$p2sPy = New-Env $Pose2SimVenv
if ($p2sPy -and -not (Test-Imports $p2sPy "Pose2Sim, rtmlib, cv2, flask, imageio_ffmpeg")) {
    Say "Installing Pose2Sim into $Pose2SimVenv (exact versions of two_view\requirements_pose2sim.txt) ..."
    & $p2sPy -m pip install -r (Join-Path $repo "two_view\requirements_pose2sim.txt")
    if (-not (Test-Imports $p2sPy "Pose2Sim, rtmlib, cv2, flask, imageio_ffmpeg")) { $failed += $Pose2SimVenv }
}
if ($p2sPy) {
    # onnxruntime-gpu 1.30 ships CUDA 13 libraries (no CUDA toolkit needed), which need NVIDIA driver 580+. Both
    # packages install the same 'onnxruntime' module, so the CPU one is removed first.
    $cudaOk = "import onnxruntime as o, sys; o.preload_dlls() if hasattr(o, 'preload_dlls') else None; sys.exit(0 if 'CUDAExecutionProvider' in o.get_available_providers() else 1)"
    if ($driverMajor -ge 580) {
        & $p2sPy -c $cudaOk 2>$null
        if ($LASTEXITCODE -ne 0) {
            Say "NVIDIA GPU found: installing onnxruntime-gpu with CUDA + cuDNN for the pose model ..."
            & $p2sPy -m pip uninstall -y onnxruntime onnxruntime-gpu
            & $p2sPy -m pip install "onnxruntime-gpu[cuda,cudnn]==1.30.0"
        }
    } elseif (-not (Test-Imports $p2sPy "onnxruntime")) {
        if ($driverMajor -gt 0) { Say "NVIDIA driver $driverMajor is older than 580: the pose model runs on the CPU (update the driver, then delete $Pose2SimVenv and start again to use the GPU)." }
        & $p2sPy -m pip install "onnxruntime==1.30.0"
    }
    if (-not (Test-Imports $p2sPy "onnxruntime")) { $failed += "$Pose2SimVenv (onnxruntime)" }
}
if (-not $p2sPy) { $failed += $Pose2SimVenv }

# --- 3c. Error-model environment ------------------------------------------------------------------------------------
$modelPy = New-Env $ModelVenv
if ($modelPy -and -not (Test-Imports $modelPy "numpy, pandas, scipy, sklearn, yaml")) {
    Say "Installing the model packages into $ModelVenv ..."
    & $modelPy -m pip install -r (Join-Path $PSScriptRoot "requirements_model.txt")
}
if ($modelPy -and -not (Test-Imports $modelPy "torch")) {
    Say "Installing torch (CPU build) into $ModelVenv ..."
    & $modelPy -m pip install "torch==2.14.1" --index-url "https://download.pytorch.org/whl/cpu"
    if ($LASTEXITCODE -ne 0) { & $modelPy -m pip install torch --index-url "https://download.pytorch.org/whl/cpu" }
}
if ($modelPy -and -not (Test-Imports $modelPy "numpy, pandas, scipy, sklearn, yaml, torch")) { $failed += $ModelVenv }
if (-not $modelPy) { $failed += $ModelVenv }

if ($failed.Count -gt 0) {
    Say "Setup is incomplete: $($failed -join ', '). Check the messages above (internet connection?) and start again." "Red"
    exit 1
}
exit 0
