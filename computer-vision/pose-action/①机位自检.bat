@echo off
chcp 65001 >nul
title PoseAction 1 - Camera Check
cd /d "%~dp0"

set "PY=%~dp0.venv\Scripts\python.exe"
rem fallback: the "0 - setup" script creates .venv in this folder

if not exist "%PY%" (
    echo [ERROR] Python environment not found.
    echo         Double-click the "0 - setup" script first.
    pause
    exit /b 1
)

echo ================================================
echo   STEP 1  Camera check
echo.
echo   Stand 1.5 - 2.5 m away from the camera.
echo   Make sure your WHOLE BODY is in the frame,
echo   including knees and ankles.
echo.
echo   Why this matters:
echo   Standing and sitting look identical in the
echo   skeleton if the legs are out of frame.
echo.
echo   s = save a screenshot   q = quit
echo ================================================
echo.

"%PY%" check_camera.py %*

echo.
echo Done. If it says the camera position is OK,
echo go on with step 2.
pause
