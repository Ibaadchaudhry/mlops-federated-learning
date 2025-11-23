# deploy.ps1 - PowerShell deployment script for Federated Learning MLOps

param(
    [switch]$Help,
    [switch]$SkipDataPrep,
    [switch]$Rebuild
)

if ($Help) {
    Write-Host @"
Federated Learning MLOps Deployment Script

Usage: .\deploy.ps1 [options]

Options:
  -Help           Show this help message
  -SkipDataPrep   Skip client data preparation (if already exists)  
  -Rebuild        Force rebuild of Docker images

Examples:
  .\deploy.ps1                    # Standard deployment
  .\deploy.ps1 -SkipDataPrep      # Skip data prep
  .\deploy.ps1 -Rebuild           # Force rebuild images
"@
    exit 0
}

# Function to print colored output
function Write-Status {
    param([string]$Message)
    Write-Host "[OK] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Cyan
}

Write-Host "Deploying Federated Learning MLOps System" -ForegroundColor Magenta
Write-Host "==============================================" -ForegroundColor Magenta

# Check if Docker is running
try {
    docker info | Out-Null
    Write-Status "Docker is running"
} catch {
    Write-Error "Docker is not running. Please start Docker Desktop and try again."
    exit 1
}

# Check if client_datasets.pkl exists
if (-not (Test-Path "client_datasets.pkl") -and -not $SkipDataPrep) {
    Write-Warning "client_datasets.pkl not found. Preparing client data..."
    
    # Run data preparation
    docker-compose --profile data-prep up --build data-prep
    
    if (Test-Path "client_datasets.pkl") {
        Write-Status "Client data prepared successfully"
    } else {
        Write-Error "Failed to prepare client data"
        exit 1
    }
} elseif ($SkipDataPrep) {
    Write-Info "Skipping data preparation (as requested)"
} else {
    Write-Status "Client data already exists"
}

# Check if models exist
if (-not (Test-Path "models") -or -not (Get-ChildItem "models" -ErrorAction SilentlyContinue)) {
    Write-Warning "No trained models found. You may need to run federated learning first."
    New-Item -ItemType Directory -Force -Path "models" | Out-Null
}

# Build and start services
Write-Host ""
Write-Host "Building Docker images..." -ForegroundColor Yellow

if ($Rebuild) {
    docker-compose build --no-cache
} else {
    docker-compose build
}

Write-Status "Docker images built successfully"

Write-Host ""
Write-Host "Starting services..." -ForegroundColor Yellow
docker-compose up -d

# Wait for services to be healthy
Write-Host ""
Write-Host "Waiting for services to be ready..." -ForegroundColor Yellow

# Function to wait for service health
function Wait-ForService {
    param(
        [string]$ServiceName,
        [string]$HealthUrl,
        [int]$MaxAttempts = 30
    )
    
    Write-Host "Waiting for $ServiceName..." -NoNewline
    
    for ($i = 1; $i -le $MaxAttempts; $i++) {
        try {
            $response = Invoke-WebRequest -Uri $HealthUrl -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
            if ($response.StatusCode -eq 200) {
                Write-Host ""
                Write-Status "$ServiceName is ready"
                return $true
            }
        } catch {
            # Service not ready yet
        }
        
        Write-Host "." -NoNewline
        Start-Sleep -Seconds 2
    }
    
    Write-Host ""
    Write-Error "$ServiceName failed to become ready"
    return $false
}

# Wait for API service
if (Wait-ForService "Model API" "http://localhost:8000/health") {
} else {
    Write-Error "Model API service failed to start"
    docker-compose logs model-api
}

# Wait for dashboard
if (Wait-ForService "Dashboard" "http://localhost:8501/_stcore/health") {
} else {
    Write-Warning "Dashboard service may not be fully ready"
}

# Show service status
Write-Host ""
Write-Host "Service Status:" -ForegroundColor Cyan
Write-Host "==================" -ForegroundColor Cyan
docker-compose ps

# Show access URLs
Write-Host ""
Write-Host "Access URLs:" -ForegroundColor Cyan
Write-Host "===============" -ForegroundColor Cyan
Write-Host "Dashboard:      http://localhost:8501"
Write-Host "Model API:      http://localhost:8000"
Write-Host "API Docs:       http://localhost:8000/docs"
Write-Host ""

# Test API endpoint
Write-Host "Testing API endpoint..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 10
    if ($response.StatusCode -eq 200) {
        Write-Status "API is responding"
        
        try {
            $modelInfo = Invoke-WebRequest -Uri "http://localhost:8000/model-info" -UseBasicParsing -TimeoutSec 5
            if ($modelInfo.StatusCode -eq 200) {
                Write-Host "Current model info:"
                $modelInfo.Content | ConvertFrom-Json | ConvertTo-Json -Depth 3
            }
        } catch {
        }
    }
} catch {
    Write-Warning "API not responding yet (may need more time)"
}

Write-Host ""
Write-Status "Deployment completed!"

Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "==============" -ForegroundColor Cyan
Write-Host "1. Open dashboard: http://localhost:8501"
Write-Host "2. Check API docs: http://localhost:8000/docs"
Write-Host "3. Run federated learning if no models exist:"
Write-Host "   docker-compose exec fl-server python fl_server.py"
Write-Host "4. Monitor logs with: docker-compose logs -f"
Write-Host ""

Write-Host "Management Commands:" -ForegroundColor Cyan
Write-Host "======================" -ForegroundColor Cyan
Write-Host "Stop services:    docker-compose down"
Write-Host "View logs:        docker-compose logs -f [service]"
Write-Host "Restart service:  docker-compose restart [service]"
Write-Host "Update code:      docker-compose up --build -d"
Write-Host ""

Write-Status "Ready for MLOps!"
