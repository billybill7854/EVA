# start.ps1 — activate .venv and launch EVA's UI (Windows).
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)
if (-not (Test-Path ".venv")) {
    throw "No .venv found. Run install.ps1 first."
}
. ".\.venv\Scripts\Activate.ps1"
$cmd = if ($args.Count -eq 0) { @("ui") } else { $args }
python run.py @cmd
