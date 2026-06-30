@echo off
setlocal enabledelayedexpansion
title BYOAI - Stopping
color 0E

echo.
echo  ============================================
echo    BYOAI - Conversational AI System
echo    STOPPING SERVICES
echo  ============================================
echo.

:: Check if we're in the right directory
if not exist "docker-compose.yml" (
    cd /d "c:\Dev\BYOAI" 2>nul
    if not exist "docker-compose.yml" (
        color 0C
        echo  [ERROR] Cannot find docker-compose.yml
        echo  Please run this script from the BYOAI project directory.
        echo.
        pause
        exit /b 1
    )
)

echo [1/3] Stopping all containers...
echo.
docker compose -f docker-compose.yml -f docker-compose.gpu.yml down 2>nul
docker compose down 2>nul
echo.

echo [2/3] Verifying all containers are stopped...
echo.
docker ps --filter "name=byoai" --format "{{.Names}}" 2>nul | findstr "byoai" >nul 2>&1
if !errorlevel! equ 0 (
    echo  [WARNING] Some containers are still running. Force stopping...
    docker stop byoai-gateway byoai-ml-service byoai-mlflow 2>nul
    docker rm byoai-gateway byoai-ml-service byoai-mlflow 2>nul
) else (
    echo  [OK] All containers stopped.
)
echo.

echo [3/3] Final status:
echo.
docker ps --filter "name=byoai" --format "table {{.Names}}\t{{.Status}}"
if !errorlevel! neq 0 (
    echo  No BYOAI containers running.
)

color 0A
echo.
echo  ============================================
echo    ALL SERVICES STOPPED
echo  ============================================
echo.
echo  To start again:  run.bat
echo.
pause
endlocal
