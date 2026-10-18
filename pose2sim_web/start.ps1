# start.ps1 -- starts Pose2Sim Web.   Right-click > Run with PowerShell, or from PowerShell:  .\pose2sim_web\start.ps1
# Uses the Pose2Sim environment C:\Pose2SimVenv (the same one as two_view\start_batch.ps1).
$py = "C:\Pose2SimVenv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "C:\Pose2SimVenv not found. Create it first (see the plan)." -ForegroundColor Red; exit 1 }
& $py -c "import Pose2Sim, flask" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Pose2Sim or Flask is not installed in C:\Pose2SimVenv. Install once with:" -ForegroundColor Yellow
    Write-Host "  $py -m pip install -r `"$(Split-Path -Parent $PSScriptRoot)\two_view\requirements_pose2sim.txt`""
    exit 1
}
Start-Job { Start-Sleep -Seconds 3; Start-Process "http://127.0.0.1:5100" } | Out-Null
Write-Host "Pose2Sim Web at http://127.0.0.1:5100  (keep this window open; Ctrl+C stops it)" -ForegroundColor Green
& $py "$PSScriptRoot\app.py"
