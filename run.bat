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

:: Start services
echo [1/3] Starting all services...
echo.
docker compose up -d
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
echo [2/3] Waiting for ML model to load (this takes ~3 minutes on first run)...
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

:: Check if gateway health endpoint responds (uses PowerShell for reliability)
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
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

color 0A
echo.
echo  ============================================
echo    ALL SERVICES RUNNING!
echo  ============================================
echo.
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
