@echo off
title BYOAI - Install
color 0B

echo.
echo  ============================================
echo    BYOAI - Conversational AI System
echo    INSTALLER
echo  ============================================
echo.

:: Check Docker
echo [1/3] Checking Docker...
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    color 0C
    echo  [ERROR] Docker is not installed!
    echo  Please install Docker Desktop from https://www.docker.com/products/docker-desktop
    echo.
    pause
    exit /b 1
)

docker info >nul 2>&1
if %errorlevel% neq 0 (
    color 0E
    echo  [WARNING] Docker Desktop is not running!
    echo  Please start Docker Desktop and try again.
    echo.
    pause
    exit /b 1
)
echo  [OK] Docker is running.
echo.

:: Check Docker Compose
echo [2/3] Checking Docker Compose...
docker compose version >nul 2>&1
if %errorlevel% neq 0 (
    color 0C
    echo  [ERROR] Docker Compose is not available!
    echo  Please update Docker Desktop to the latest version.
    echo.
    pause
    exit /b 1
)
echo  [OK] Docker Compose is available.
echo.

:: Build images
echo [3/3] Building Docker images (this may take a few minutes)...
echo.
docker compose build
if %errorlevel% neq 0 (
    color 0C
    echo.
    echo  [ERROR] Build failed! Check the errors above.
    echo.
    pause
    exit /b 1
)

color 0A
echo.
echo  ============================================
echo    INSTALLATION COMPLETE!
echo  ============================================
echo.
echo  To start the system, run:  run.bat
echo.
pause
