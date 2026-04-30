# install.ps1 — set up EVA on Windows (PowerShell 5.1 / 7+).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File install.ps1
#   powershell -ExecutionPolicy Bypass -File install.ps1 -Gpu
#   powershell -ExecutionPolicy Bypass -File install.ps1 -NoUi
#
# After install:
#   .\.venv\Scripts\Activate.ps1
#   python run.py ui

[CmdletBinding()]
param(
    [switch]$Gpu,
    [switch]$NoUi
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)

$python = if (Get-Command python3 -ErrorAction SilentlyContinue) { "python3" }
          elseif (Get-Command python -ErrorAction SilentlyContinue) { "python" }
          else { throw "Python 3.10+ not found on PATH." }

$pyVer = & $python -c "import sys; print('%d.%d' % sys.version_info[:2])"
if ($pyVer -notin "3.10", "3.11", "3.12", "3.13") {
    throw "EVA requires Python 3.10+ (found $pyVer)."
}

Write-Host "[eva] Creating virtual environment in .venv (Python $pyVer)"
& $python -m venv .venv

$activate = Join-Path $PWD ".venv\Scripts\Activate.ps1"
. $activate

python -m pip install --upgrade pip wheel setuptools | Out-Null

if ($Gpu) {
    Write-Host "[eva] Installing CUDA torch."
    pip install --index-url https://download.pytorch.org/whl/cu121 torch
} else {
    Write-Host "[eva] Installing CPU torch."
    pip install torch
}

$extra = ""
if (-not $NoUi) { $extra = "[ui]" }

Write-Host "[eva] Installing EVA (editable, extras=$extra)."
pip install -e ".$extra"

if (Test-Path ".pre-commit-config.yaml") {
    Write-Host "[eva] Configuring pre-commit hooks."
    pip install pre-commit | Out-Null
    pre-commit install | Out-Null
}

Write-Host ""
Write-Host "[eva] Install complete."
Write-Host "     .\.venv\Scripts\Activate.ps1"
Write-Host "     python run.py ui     # open http://127.0.0.1:8765"
