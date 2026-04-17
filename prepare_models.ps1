$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

$env:MODEL_AUTO_DOWNLOAD = "1"
$env:MODEL_DOWNLOAD_QUIET = "0"
python -c "import main; main.download_models(); print('Models prepared successfully')"
