# start_website.ps1 -- one-click launcher for the Squat Studio website.
#
#   Right-click > Run with PowerShell   (or:  powershell -ExecutionPolicy Bypass -File two_view\start_website.ps1)
#
# First run: creates the Python environment (C:\Proj1Venv) and installs the packages (~10 minutes, needs internet).
# Later runs: just starts the website and opens it in your browser.
# The environment lives at C:\Proj1Venv (not inside the repo) because this repo's folder name is so long that
# Windows can hit its path-length limit when installing big packages.

$ErrorActionPreference = "Stop"
$venv = "C:\Proj1Venv"
$py   = "$venv\Scripts\python.exe"
$repo = Split-Path -Parent $PSScriptRoot

if (-not (Test-Path $py)) {
    Write-Host "First run: creating the environment at $venv ..." -ForegroundColor Yellow
    py -3.11 -m venv $venv
    & $py -m pip install --upgrade pip
    & $py -m pip install -r "$repo\requirements_updated_v2.txt" imageio-ffmpeg
}

Write-Host "Starting Squat Studio at http://127.0.0.1:5000  (close this window or press Ctrl+C to stop)" -ForegroundColor Green
Start-Job { Start-Sleep -Seconds 4; Start-Process "http://127.0.0.1:5000" } | Out-Null
Set-Location $repo
& $py "$repo\two_view\web_app.py"
