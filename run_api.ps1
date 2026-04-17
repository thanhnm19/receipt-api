$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

$env:MODEL_AUTO_DOWNLOAD = "0"
uvicorn main:app --host 0.0.0.0 --port 8000
