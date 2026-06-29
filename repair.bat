@echo off
setlocal enabledelayedexpansion
title BYOAI - Repair
color 0E

echo.
echo  ============================================
echo    BYOAI - Conversational AI System
echo    REPAIR / REBUILD
echo  ============================================
echo.
echo  This will stop all services, rebuild from
echo  scratch, and restart everything.
echo.
echo  Press any key to continue or Ctrl+C to cancel...
pause >nul
echo.

:: Check Docker
docker info >nul 2>&1
if !errorlevel! neq 0 (
    color 0C
    echo  [ERROR] Docker Desktop is not running!
    echo  Please start Docker Desktop and try again.
    echo.
    pause
    exit /b 1
)

:: Stop everything (both CPU and GPU compose files)
echo [1/4] Stopping all services...
docker compose -f docker-compose.yml -f docker-compose.gpu.yml down --remove-orphans 2>nul
docker compose down --remove-orphans 2>nul
echo  [OK] Services stopped.
echo.

:: Remove old images
echo [2/4] Removing old images...
docker compose -f docker-compose.yml -f docker-compose.gpu.yml down --rmi local --volumes 2>nul
docker compose down --rmi local --volumes 2>nul
echo  [OK] Old images removed.
echo.

:: Rebuild
echo [3/4] Rebuilding all images (no cache)...
echo.
docker compose build --no-cache
if !errorlevel! neq 0 (
    color 0C
    echo.
    echo  [ERROR] Build failed! Check the errors above.
    echo.
    pause
    exit /b 1
)
echo.
echo  [OK] Build complete.
echo.

:: Restart
echo [4/4] Starting services...
echo.
call run.bat
endlocal
