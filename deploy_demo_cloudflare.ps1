param(
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue)) {
    Write-Host "cloudflared not found." -ForegroundColor Red
    Write-Host "Install here: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/" -ForegroundColor Yellow
    throw "Missing cloudflared CLI"
}

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

$pythonExe = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

& $pythonExe -m pip install --upgrade pip
& $pythonExe -m pip install -r requirements.txt

$merchantModel = Join-Path $PSScriptRoot "models\merchant_classifier_latest.pkl"
$lineitemModel = Join-Path $PSScriptRoot "models\lineitem_classifier_latest.pkl"

if ((Test-Path $merchantModel) -and (Test-Path $lineitemModel)) {
    $env:MODEL_AUTO_DOWNLOAD = "0"
} else {
    $env:MODEL_AUTO_DOWNLOAD = "1"
}
$env:MODEL_DOWNLOAD_QUIET = "1"

Write-Host "Starting local API at http://127.0.0.1:$Port ..."
$apiProcess = Start-Process -FilePath $pythonExe `
    -ArgumentList "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$Port" `
    -WorkingDirectory $PSScriptRoot `
    -PassThru

$tunnelProcess = $null

try {
    Write-Host "Creating public URL with Cloudflare Tunnel..."

    $stdoutFile = Join-Path $PSScriptRoot ".cloudflared.stdout.log"
    $stderrFile = Join-Path $PSScriptRoot ".cloudflared.stderr.log"
    if (Test-Path $stdoutFile) { Remove-Item $stdoutFile -Force }
    if (Test-Path $stderrFile) { Remove-Item $stderrFile -Force }

    $tunnelProcess = Start-Process -FilePath "cloudflared" `
        -ArgumentList "tunnel", "--url", "http://127.0.0.1:$Port", "--no-autoupdate" `
        -WorkingDirectory $PSScriptRoot `
        -RedirectStandardOutput $stdoutFile `
        -RedirectStandardError $stderrFile `
        -PassThru

    $deadline = (Get-Date).AddSeconds(45)
    $publicUrl = $null
    while ((Get-Date) -lt $deadline -and -not $publicUrl) {
        foreach ($logFile in @($stdoutFile, $stderrFile)) {
            if (Test-Path $logFile) {
                $content = Get-Content $logFile -Raw
                if (-not [string]::IsNullOrWhiteSpace($content)) {
                    $m = [regex]::Match($content, "https://[a-zA-Z0-9-]+\.trycloudflare\.com")
                    if ($m.Success) {
                        $publicUrl = $m.Value
                        break
                    }
                }
            }
        }
        if ($publicUrl) { break }
        Start-Sleep -Milliseconds 500
    }

    if ($publicUrl) {
        Write-Host "PUBLIC BASE_URL: $publicUrl" -ForegroundColor Green
        Write-Host "Health: $publicUrl/health"
        Write-Host "Model status: $publicUrl/model/status"
        Write-Host "Receipt endpoint: $publicUrl/receipt"
    } else {
        Write-Host "Could not auto-detect trycloudflare URL. Check logs below:" -ForegroundColor Yellow
    }

    Write-Host "Tunnel is running. Press Ctrl+C to stop."
    if (Test-Path $stderrFile) {
        Get-Content $stderrFile -Wait
    } elseif (Test-Path $stdoutFile) {
        Get-Content $stdoutFile -Wait
    } else {
        Wait-Process -Id $tunnelProcess.Id
    }
}
finally {
    if ($tunnelProcess -and -not $tunnelProcess.HasExited) {
        Stop-Process -Id $tunnelProcess.Id -Force
    }
    if ($apiProcess -and -not $apiProcess.HasExited) {
        Stop-Process -Id $apiProcess.Id -Force
    }
}
