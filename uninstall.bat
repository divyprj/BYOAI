@echo off
title BYOAI - Uninstall
color 0C

echo.
echo  ============================================
echo    BYOAI - Conversational AI System
echo    UNINSTALLER
echo  ============================================
echo.
echo  This will completely remove:
echo    - All running containers
echo    - All Docker images (gateway, ml-service, mlflow)
echo    - All Docker volumes (model cache, mlflow data)
echo.
echo  Your source code will NOT be deleted.
echo.

set /p confirm="Are you sure? Type YES to confirm: "
if /i not "%confirm%"=="YES" (
    echo.
    echo  Uninstall cancelled.
    echo.
    pause
    exit /b 0
)

echo.

:: Stop containers
echo [1/4] Stopping all containers...
docker compose down --remove-orphans 2>nul
echo  [OK] Containers stopped.
echo.

:: Remove images
echo [2/4] Removing Docker images...
docker compose down --rmi all 2>nul
echo  [OK] Images removed.
echo.

:: Remove volumes
echo [3/4] Removing Docker volumes (model cache, mlflow data)...
docker volume rm byoi_model-cache 2>nul
docker volume rm byoi_mlflow-data 2>nul
echo  [OK] Volumes removed.
echo.

:: Clean up dangling resources
echo [4/4] Cleaning up dangling resources...
docker system prune -f >nul 2>&1
echo  [OK] Cleanup complete.

color 0A
echo.
echo  ============================================
echo    UNINSTALL COMPLETE!
echo  ============================================
echo.
echo  All Docker resources have been removed.
echo  Your source code is still in this folder.
echo.
echo  To reinstall, run:  install.bat
echo.
pause
