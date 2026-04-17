param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectId,

    [string]$Region = "asia-southeast1",
    [string]$ServiceName = "receipt-ml-api",
    [int]$TimeoutSeconds = 900,
    [int]$MinInstances = 1,
    [int]$MaxInstances = 3
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    throw "gcloud CLI not found. Install Google Cloud SDK first: https://cloud.google.com/sdk/docs/install"
}

Write-Host "[1/5] Set active project..."
gcloud config set project $ProjectId | Out-Null

Write-Host "[Precheck] Validate billing is enabled for project..."
$billingEnabled = gcloud billing projects describe $ProjectId --format="value(billingEnabled)"
if ($billingEnabled -ne "True") {
    Write-Host "Billing is disabled for project: $ProjectId" -ForegroundColor Red
    Write-Host "Open: https://console.cloud.google.com/billing/linkedaccount?project=$ProjectId" -ForegroundColor Yellow
    Write-Host "After linking a billing account, rerun this script." -ForegroundColor Yellow
    throw "Billing must be enabled before Cloud Run deployment."
}

Write-Host "[2/5] Enable required APIs..."
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

$image = "gcr.io/$ProjectId/$ServiceName"

Write-Host "[3/5] Build image (this may take a few minutes because models are baked into image)..."
gcloud builds submit --tag $image .

Write-Host "[4/5] Deploy Cloud Run service..."
gcloud run deploy $ServiceName `
    --image $image `
    --region $Region `
    --platform managed `
    --allow-unauthenticated `
    --port 8080 `
    --cpu 1 `
    --memory 1Gi `
    --timeout $TimeoutSeconds `
    --concurrency 40 `
    --min-instances $MinInstances `
    --max-instances $MaxInstances `
    --set-env-vars MODEL_AUTO_DOWNLOAD=0,MODEL_DOWNLOAD_QUIET=1

$url = gcloud run services describe $ServiceName --region $Region --format='value(status.url)'

Write-Host "[5/5] Done"
Write-Host "API URL: $url"
Write-Host "Health check: $url/health"
Write-Host "Model status: $url/model/status"
Write-Host "Receipt endpoint: $url/receipt"
