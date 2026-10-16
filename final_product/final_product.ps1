# final_product.ps1 -- one-click launcher for the final product: upload a front + side squat video, get the errors
# per rep, the 3D body with the error's saliency, the keypoint videos and the rep table on one timeline.
#
#   Right-click > Run with PowerShell   (or:  powershell -ExecutionPolicy Bypass -File final_product\final_product.ps1)
#
# Opens http://127.0.0.1:5090 . KEEP THIS WINDOW OPEN while an analysis runs (closing it stops the analysis;
# press Retry on the session afterwards to continue).
#
# Needs the two environments the other launchers create:
#   C:\Proj1Venv      (start_website.ps1)  - this app, the 3D body mesh (MotionBERT + SMPL)
#   C:\Pose2SimVenv   (start_batch.ps1)    - sync, calibration, 2D keypoints, 3D triangulation
# and the SMPL body model files in MotionBERT\data\mesh\ for the 3D body (licence-gated, not in git).
# The error-detection model is used once it is trained and exported to final_product\squat_model_export\ (model.pt) with
# its environment C:\SquatModelVenv; until then the sessions show the reps without errors (demo predictions can be
# switched on per session to try the page).
#
# Options:  -PoseMode lightweight|balanced|performance   (default performance = the same as the dataset batch;
#                                                          lightweight is ~3x faster on a CPU but less accurate)
#           -Device auto|cpu|cuda                        (default auto)
#           -Port 5090

param(
    [ValidateSet("lightweight", "balanced", "performance")][string]$PoseMode = "performance",
    [ValidateSet("auto", "cpu", "cuda")][string]$Device = "auto",
    [int]$Port = 5090
)
# "Continue": in Windows PowerShell 5.1 "Stop" turns any stderr line of python/pip into a fatal error.
$ErrorActionPreference = "Continue"
$py   = "C:\Proj1Venv\Scripts\python.exe"
$p2s  = "C:\Pose2SimVenv\Scripts\python.exe"
$repo = Split-Path -Parent $PSScriptRoot

if (-not (Test-Path $py)) {
    Write-Host "C:\Proj1Venv is missing. Run two_view\start_website.ps1 once to create it, then run this again." -ForegroundColor Red
    Read-Host "Press Enter to close"; exit 1
}
& $py -c "import flask, numpy, cv2" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "C:\Proj1Venv is incomplete (flask / numpy / opencv). Run two_view\start_website.ps1 again." -ForegroundColor Red
    Read-Host "Press Enter to close"; exit 1
}
& $py -c "import smplx, torch" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing smplx into C:\Proj1Venv (needed for the 3D body) ..." -ForegroundColor Yellow
    & $py -m pip install smplx
}
if (-not (Test-Path $p2s)) {
    Write-Host "WARNING: C:\Pose2SimVenv is missing - analyses cannot run until it exists." -ForegroundColor Red
    Write-Host "         Run two_view\start_batch.ps1 once (it creates the environment), then start this again." -ForegroundColor Red
}
$smpl = Join-Path $repo "MotionBERT\data\mesh\SMPL_NEUTRAL.pkl"
if (-not (Test-Path $smpl)) {
    Write-Host "WARNING: $smpl is missing - the 3D body cannot be built (everything else works)." -ForegroundColor Red
}
if (-not (Test-Path (Join-Path $repo "final_product\squat_model_export\model.pt"))) {
    Write-Host "Note: the error-detection model is not trained yet - reps are shown without errors (try the demo predictions)." -ForegroundColor Yellow
}

$env:PRODUCT_POSE_MODE = $PoseMode
$env:PRODUCT_DEVICE = $Device
$env:PRODUCT_PORT = "$Port"
Write-Host "Starting Squat Analysis at http://127.0.0.1:$Port  (pose model: $PoseMode, device: $Device)" -ForegroundColor Green
Write-Host "Keep this window open while analyses run; close it or press Ctrl+C to stop." -ForegroundColor Green
Start-Job -ArgumentList $Port { param($p) Start-Sleep -Seconds 4; Start-Process "http://127.0.0.1:$p" } | Out-Null
Set-Location $repo
& $py "$repo\final_product\product_app.py"
