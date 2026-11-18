@echo off
chcp 65001 >nul
title PoseAction 5 - Report
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
echo   Offline analysis for every recorded clip
echo   Output: three-panel charts in reports\
echo ================================================
echo.

"%PY%" report.py --all

echo.
echo Opening reports folder...
start "" "%~dp0reports"
pause
