# setup_env.ps1 -- creates the model environment C:\SquatModelVenv used by the research notebooks and by the final
# product's error-detection step.
#
#   powershell -ExecutionPolicy Bypass -File research\setup_env.ps1            (GPU build when an NVIDIA GPU is present)
#   powershell -ExecutionPolicy Bypass -File research\setup_env.ps1 -ForceCpu
#
# torch: CUDA 13.0 build when the NVIDIA driver is >= 580 (needed for RTX 50-series / Blackwell), CUDA 12.6 build for
# driver >= 525, CPU build without an NVIDIA GPU. Only the driver is needed, no CUDA toolkit.
# Also registers the Jupyter kernel "Python (SquatModelVenv)" that the notebooks use.

param([switch]$ForceCpu, [switch]$ReinstallTorch)
$ErrorActionPreference = "Continue"
$venv = "C:\SquatModelVenv"
$py = "$venv\Scripts\python.exe"
$req = Join-Path $PSScriptRoot "requirements.txt"
$torchVer = "2.14.1"

if (-not (Test-Path $py)) {
    & py -3.11 -c "import sys" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Python 3.11 not found; installing it with winget ..." -ForegroundColor Yellow
        winget install --id Python.Python.3.11 -e --accept-source-agreements --accept-package-agreements
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
    }
    Write-Host "Creating $venv ..." -ForegroundColor Yellow
    py -3.11 -m venv $venv
    & $py -m pip install --upgrade pip
}
& $py -m pip install -r $req
if ($LASTEXITCODE -ne 0) { throw "Installing the packages failed (see above)." }

$want = "cpu"
if (-not $ForceCpu -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    $drv = (nvidia-smi --query-gpu=driver_version --format=csv,noheader | Select-Object -First 1).Trim()
    $cc = (nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null | Select-Object -First 1)
    $major = [int]($drv.Split('.')[0])
    $want = if ($major -ge 580) { "cu130" } else { "cu126" }
    if ($cc -and [double]$cc.Trim() -ge 12.0 -and $major -lt 580) {
        Write-Host "WARNING: this RTX 50-series GPU needs NVIDIA driver 580+ for the CUDA 13 torch build; driver $drv is older, so training would run on the CPU." -ForegroundColor Red
    }
}
& $py -c "import torch, sys; v = torch.version.cuda; sys.exit(0 if ('cpu' if v is None else 'cu' + v.replace('.', '')) == '$want' else 3)" 2>$null
if ($ReinstallTorch -or $LASTEXITCODE -ne 0) {
    & $py -m pip uninstall -y torch 2>$null
    Write-Host "Installing torch $torchVer ($want) ..." -ForegroundColor Yellow
    & $py -m pip install "torch==$torchVer" --index-url "https://download.pytorch.org/whl/$want"
    if ($LASTEXITCODE -ne 0) { & $py -m pip install torch --index-url "https://download.pytorch.org/whl/$want" }
}
& $py -m ipykernel install --user --name squatmodel --display-name "Python (SquatModelVenv)"
& $py -c "import torch; ok = torch.cuda.is_available(); print('torch', torch.__version__, '->', ('GPU ' + torch.cuda.get_device_name(0)) if ok else 'CPU')"
Write-Host "Done. Open research\notebooks in Jupyter / VS Code with the kernel 'Python (SquatModelVenv)'." -ForegroundColor Green
