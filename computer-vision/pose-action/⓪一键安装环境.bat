@echo off
chcp 65001 >nul
title PoseAction 0 - Setup Environment
cd /d "%~dp0"

echo ==========================================================
echo   PoseAction - one-time environment setup
echo.
echo   Needs internet. Downloads ~500 MB (PyTorch is the big one).
echo   Usually takes 3-10 minutes.
echo ==========================================================
echo.

if exist "%~dp0.venv\Scripts\python.exe" (
    echo [SKIP] .venv already exists in this folder - nothing to do.
    echo        Delete the .venv folder if you want a clean reinstall.
    echo.
    pause
    exit /b 0
)

rem ---------- locate a Python interpreter ----------
set "BASEPY="

py -3.11 -c "import sys" >nul 2>&1
if not errorlevel 1 set "BASEPY=py -3.11"

if not defined BASEPY (
    py -3 -c "import sys" >nul 2>&1
    if not errorlevel 1 set "BASEPY=py -3"
)

if not defined BASEPY (
    python -c "import sys" >nul 2>&1
    if not errorlevel 1 set "BASEPY=python"
)

if not defined BASEPY (
    echo [ERROR] Python not found on this machine.
    echo.
    echo   1. Download Python 3.11: https://www.python.org/downloads/
    echo   2. During install, TICK "Add python.exe to PATH"
    echo   3. Run this script again.
    echo.
    pause
    exit /b 1
)

echo [1/3] Python found:
%BASEPY% --version
echo.

echo [2/3] Creating virtual environment .venv ...
%BASEPY% -m venv ".venv"
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo.
    echo [ERROR] Failed to create .venv
    echo         Try running this script as administrator.
    pause
    exit /b 1
)
echo       OK
echo.

echo [3/3] Installing dependencies ... this is the slow part.
"%~dp0.venv\Scripts\python.exe" -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
"%~dp0.venv\Scripts\python.exe" -m pip install -r "%~dp0requirements-lock.txt" -i https://pypi.tuna.tsinghua.edu.cn/simple

if errorlevel 1 (
    echo.
    echo [WARN] Mirror failed, retrying with official PyPI ...
    "%~dp0.venv\Scripts\python.exe" -m pip install -r "%~dp0requirements-lock.txt"
)

echo.
echo Verifying imports ...
"%~dp0.venv\Scripts\python.exe" -c "import cv2,torch,ultralytics,pandas,sklearn,matplotlib,openpyxl,joblib;print('All imports OK')"
if errorlevel 1 (
    echo.
    echo [ERROR] Some packages failed to install.
    echo         Run this script again, or check your network.
    pause
    exit /b 1
)

echo.
echo ==========================================================
echo   Setup finished successfully.
echo   Now double-click the record script to start.
echo ==========================================================
pause
