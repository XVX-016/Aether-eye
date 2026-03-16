$ErrorActionPreference = "Stop"

Write-Host "Starting Aether-Eye demo stack..."
docker compose up -d

$backendHealthy = $false
for ($i = 0; $i -lt 60; $i++) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
            $backendHealthy = $true
            break
        }
    } catch {
        Start-Sleep -Seconds 2
    }
}

if (-not $backendHealthy) {
    throw "Backend health check failed."
}

python scripts/seed_demo_data.py

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "AETHER-EYE  //  OPERATIONAL" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Dashboard:  http://localhost:3000"
Write-Host "API:        http://localhost:8000/docs"
Write-Host "Sites:      18 monitored globally"
Write-Host "==========================================" -ForegroundColor Cyan
