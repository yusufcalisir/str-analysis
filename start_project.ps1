$ErrorActionPreference = "Stop"
Write-Host "VANTAGE-STR: Starting Development Environment..." -ForegroundColor Green

# Ensure we are in the project root
$rootDir = $PSScriptRoot
Set-Location "$rootDir"

# 1. Start Infrastructure
Write-Host "`n[1/3] Starting Infrastructure..." -ForegroundColor Cyan
try {
    # Check if we need to remove stopped containers first? Maybe.
    docker-compose -f "infra\docker-compose.yml" up -d
    Write-Host "Infrastructure running." -ForegroundColor Green
}
catch {
    Write-Host "Docker failed. Ensure Docker Desktop is running." -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    exit 1
}

# 2. Start Backend
Write-Host "`n[2/3] Starting Backend..." -ForegroundColor Cyan
try {
    $backendPath = Join-Path "$rootDir" "backend"
    Start-Process -FilePath "powershell.exe" -WorkingDirectory "$backendPath" -ArgumentList "-NoExit", "-Command", "& { .\venv\Scripts\activate; uvicorn app.main:app --reload --port 8000 }"
    Write-Host "Backend started." -ForegroundColor Green
}
catch {
    Write-Host "Failed to start Backend: $_" -ForegroundColor Red
}

# 3. Start Frontend
Write-Host "`n[3/3] Starting Frontend..." -ForegroundColor Cyan
try {
    $frontendPath = Join-Path "$rootDir" "frontend"
    Start-Process -FilePath "powershell.exe" -WorkingDirectory "$frontendPath" -ArgumentList "-NoExit", "-Command", "npm run dev"
    Write-Host "Frontend started." -ForegroundColor Green
}
catch {
    Write-Host "Failed to start Frontend: $_" -ForegroundColor Red
}

Write-Host "`nVANTAGE-STR is now running!" -ForegroundColor Yellow
Write-Host "   Frontend: http://localhost:3000"
Write-Host "   Backend:  http://localhost:8000/docs"
