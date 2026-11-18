@echo off
chcp 65001 >nul
title PoseAction - Environment Check
cd /d "%~dp0"

set "PY=%~dp0.venv\Scripts\python.exe"
rem fallback: the "0 - setup" script creates .venv in this folder

if not exist "%PY%" (
    echo.
    echo [ERROR] Python environment not found.
    echo         Double-click "0-setup" script first.
    echo.
    pause
    exit /b 1
)

"%PY%" check_env.py
