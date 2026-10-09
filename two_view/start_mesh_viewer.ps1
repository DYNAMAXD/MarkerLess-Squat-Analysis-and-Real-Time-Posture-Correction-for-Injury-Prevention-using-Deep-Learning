# start_mesh_viewer.ps1 -- opens the 3D mesh + saliency viewer (two_view\mesh_app.py) at http://127.0.0.1:5070
#
#   Right-click > Run with PowerShell   (or:  powershell -ExecutionPolicy Bypass -File two_view\start_mesh_viewer.ps1)
#
# Needs the project environment C:\Proj1Venv (made by start_website.ps1; it has torch + smplx) and the SMPL body
# model files in MotionBERT\data\mesh\ (SMPL_NEUTRAL.pkl etc. - licence-gated, not in git; copy them from the PC
# that has them). Works on the folders the batch wrote (<dataset>\preprocessed_data) or the website's runs
# (outputs\two_view). Building a recording's mesh takes about a minute on a CPU.

$ErrorActionPreference = "Continue"
$py   = "C:\Proj1Venv\Scripts\python.exe"
$repo = Split-Path -Parent $PSScriptRoot

if (-not (Test-Path $py)) {
    Write-Host "C:\Proj1Venv is missing. Run two_view\start_website.ps1 once to create it, then run this again." -ForegroundColor Red
    Read-Host "Press Enter to close"; exit 1
}
& $py -c "import smplx, torch, flask" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing smplx into C:\Proj1Venv ..." -ForegroundColor Yellow
    & $py -m pip install smplx
}
$smpl = Join-Path $repo "MotionBERT\data\mesh\SMPL_NEUTRAL.pkl"
if (-not (Test-Path $smpl)) {
    Write-Host "WARNING: $smpl is missing - meshes cannot be built until the SMPL files are copied there" -ForegroundColor Red
    Write-Host "         (SMPL_NEUTRAL.pkl, smpl_mean_params.npz, J_regressor_extra.npy, J_regressor_h36m_correct.npy)." -ForegroundColor Red
}

Write-Host "Starting the mesh viewer at http://127.0.0.1:5070  (close this window or press Ctrl+C to stop)" -ForegroundColor Green
Start-Job { Start-Sleep -Seconds 4; Start-Process "http://127.0.0.1:5070" } | Out-Null
Set-Location $repo
& $py "$repo\two_view\mesh_app.py"
