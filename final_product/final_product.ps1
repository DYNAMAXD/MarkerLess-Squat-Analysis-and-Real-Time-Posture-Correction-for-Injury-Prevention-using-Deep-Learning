# final_product.ps1 -- one-click launcher for the final product: upload a front + side squat video, get the errors
# per rep, the 3D body with the error's saliency, the keypoint videos and the rep table on one timeline.
#
#   Right-click > Run with PowerShell   (or:  powershell -ExecutionPolicy Bypass -File final_product\final_product.ps1)
#
# A fresh clone needs nothing else: the first start runs final_product\setup.ps1, which installs Python 3.11 if
# missing, pulls the Git LFS files and creates the three environments (needs internet, ~15-25 min, ~6 GB on C:):
#   C:\Proj1Venv       this app, the 3D body mesh (MotionBERT + SMPL)
#   C:\Pose2SimVenv    sync, calibration, 2D keypoints, 3D triangulation
#   C:\SquatModelVenv  the error-detection model (research\, CNN-LSTM, final_product\squat_model_export\)
# Later starts only check them (a few seconds) and open http://127.0.0.1:5090 . KEEP THIS WINDOW OPEN while an
# analysis runs (closing it stops the analysis; press Retry on the session afterwards to continue).
#
# Options:  -PoseMode lightweight|balanced|performance   (default performance = the same as the dataset batch;
#                                                          lightweight is ~3x faster on a CPU but less accurate)
#           -Device auto|cpu|cuda                        (default auto)
#           -Port 5090
#           -ForceCpu                                    (setup: no GPU packages even with an NVIDIA GPU)
#           -AppVenv / -Pose2SimVenv / -ModelVenv        (other environment folders)

param(
    [ValidateSet("lightweight", "balanced", "performance")][string]$PoseMode = "performance",
    [ValidateSet("auto", "cpu", "cuda")][string]$Device = "auto",
    [int]$Port = 5090,
    [switch]$ForceCpu,
    [string]$AppVenv = "C:\Proj1Venv",
    [string]$Pose2SimVenv = "C:\Pose2SimVenv",
    [string]$ModelVenv = "C:\SquatModelVenv"
)
# "Continue": in Windows PowerShell 5.1 "Stop" turns any stderr line of python/pip into a fatal error.
$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot

Write-Host "Checking the installation (the first start sets everything up; this takes a while once) ..." -ForegroundColor Green
& powershell -NoProfile -ExecutionPolicy Bypass -File "$PSScriptRoot\setup.ps1" -AppVenv $AppVenv -Pose2SimVenv $Pose2SimVenv -ModelVenv $ModelVenv -ForceCpu:$ForceCpu
$setupOk = $LASTEXITCODE -eq 0

$py = Join-Path $AppVenv "Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "The app environment $AppVenv could not be created - see the messages above." -ForegroundColor Red
    Read-Host "Press Enter to close"; exit 1
}
if (-not $setupOk) {
    Write-Host "Starting anyway; the steps that need the missing parts will report an error." -ForegroundColor Yellow
}
if (-not (Test-Path (Join-Path $repo "final_product\squat_model_export\model.pt"))) {
    Write-Host "Note: no error-detection model in final_product\squat_model_export\ - reps are shown without errors." -ForegroundColor Yellow
}

$env:POSE2SIM_PY = Join-Path $Pose2SimVenv "Scripts\python.exe"
$env:SQUAT_MODEL_PY = Join-Path $ModelVenv "Scripts\python.exe"
$env:PRODUCT_POSE_MODE = $PoseMode
$env:PRODUCT_DEVICE = $Device
$env:PRODUCT_PORT = "$Port"
Write-Host "Starting Squat Analysis at http://127.0.0.1:$Port  (pose model: $PoseMode, device: $Device)" -ForegroundColor Green
Write-Host "Keep this window open while analyses run; close it or press Ctrl+C to stop." -ForegroundColor Green
Start-Job -ArgumentList $Port { param($p) Start-Sleep -Seconds 4; Start-Process "http://127.0.0.1:$p" } | Out-Null
Set-Location $repo
& $py "$repo\final_product\product_app.py"
