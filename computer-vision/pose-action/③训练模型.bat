@echo off
chcp 65001 >nul
title PoseAction 3 - Train
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
echo   Train classifier on recorded data
echo   Needs at least 2 clips per action, 5 is better.
echo ================================================
echo.

"%PY%" train.py

echo.
echo Confusion matrix saved to: %~dp0reports
pause
