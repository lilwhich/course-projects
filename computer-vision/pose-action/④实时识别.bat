@echo off
chcp 65001 >nul
title PoseAction 4 - Realtime
cd /d "%~dp0"

set "PY=%~dp0.venv\Scripts\python.exe"
rem fallback: the "0 - setup" script creates .venv in this folder

if not exist "%PY%" (
    echo [ERROR] Python environment not found.
    echo         Double-click the "0 - setup" script first.
    pause
    exit /b 1
)
if not exist "%~dp0models\pose_best.pkl" (
    echo [WARN] No trained model found. Run step 2 first.
    pause
    exit /b 1
)

echo ================================================
echo   Realtime recognition
echo   Waits 1.5s to fill the buffer, then shows result.
echo   Press q in the video window to quit.
echo ================================================
echo.

"%PY%" realtime.py

pause
