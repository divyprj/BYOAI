@echo off
setlocal enabledelayedexpansion
title BYOAI - Running
color 0B

echo.
echo  ============================================
echo    BYOAI - Conversational AI System
echo    STARTING SERVICES
echo  ============================================
echo.

:: Check Docker is running
docker info >nul 2>&1
if !errorlevel! neq 0 (
    color 0E
    echo  [WARNING] Docker Desktop is not running!
    echo  Please start Docker Desktop and try again.
    echo.
    pause
    exit /b 1
)

:: Device selection menu
echo  Select compute device for ML inference:
echo.
echo    [1] CPU        (Default - works everywhere, ~2s per request)
echo    [2] NVIDIA GPU (Requires NVIDIA GPU + drivers, ~0.05s per request)
echo.
set /p DEVICE_CHOICE="  Enter choice [1/2] (default=1): "
if "!DEVICE_CHOICE!"=="" set DEVICE_CHOICE=1

if "!DEVICE_CHOICE!"=="2" (
    echo.
    echo  Checking NVIDIA GPU availability...
    docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi >nul 2>&1
    if !errorlevel! neq 0 (
        color 0E
        echo  [WARNING] NVIDIA GPU not available in Docker!
        echo  Make sure you have:
        echo    - NVIDIA GPU drivers installed
        echo    - Docker Desktop with WSL2 backend
        echo    - GPU support enabled in Docker Desktop settings
        echo.
        echo  Falling back to CPU mode...
        set DEVICE_CHOICE=1
        timeout /t 3 /nobreak >nul
    ) else (
        echo  [OK] NVIDIA GPU detected!
    )
)
echo.

:: Start services based on device choice
echo [1/3] Starting all services...
echo.

if "!DEVICE_CHOICE!"=="2" (
    echo  Mode: NVIDIA GPU
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
) else (
    echo  Mode: CPU
    docker compose up -d
)

if !errorlevel! neq 0 (
    color 0C
    echo.
    echo  [ERROR] Failed to start services! Run repair.bat to fix.
    echo.
    pause
    exit /b 1
)

:: Wait for ML model to load
echo.
if "!DEVICE_CHOICE!"=="2" (
    echo [2/3] Waiting for ML model to load on GPU...
) else (
    echo [2/3] Waiting for ML model to load (this takes ~3 minutes on first run)...
)
echo.
set ATTEMPTS=0

:wait_loop
set /a ATTEMPTS=ATTEMPTS+1
if !ATTEMPTS! gtr 60 (
    color 0E
    echo.
    echo  [WARNING] Services are taking longer than expected.
    echo  Check status with:  docker ps
    echo  View logs with:     docker compose logs -f
    echo.
    pause
    exit /b 1
)

:: Check if gateway health endpoint responds
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri 'http://localhost:8000/health' -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop; exit 0 } catch { exit 1 }" >nul 2>&1
if !errorlevel! neq 0 (
    echo  Waiting... [!ATTEMPTS!/60] ^(checking every 5 seconds^)
    timeout /t 5 /nobreak >nul
    goto wait_loop
)

:: All services are up
echo  [OK] All services are healthy!
echo.

:: Show status
echo [3/3] Service Status:
echo.
if "!DEVICE_CHOICE!"=="2" (
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
) else (
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
)

color 0A
echo.
echo  ============================================
echo    ALL SERVICES RUNNING!
echo  ============================================
echo.
if "!DEVICE_CHOICE!"=="2" (
    echo  Compute:        NVIDIA GPU
) else (
    echo  Compute:        CPU
)
echo  API Gateway:    http://localhost:8000
echo  Swagger UI:     http://localhost:8000/docs
echo  ML Service:     http://localhost:8001
echo  MLflow:         http://localhost:5000
echo.
echo  To stop:  docker compose down
echo.

:: Open browser
echo Opening Swagger UI in browser...
start http://localhost:8000/docs
echo.
pause
endlocal
