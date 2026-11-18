@echo off
chcp 65001 >nul
title PoseAction 2 - Record
cd /d "%~dp0"

rem prefer the local .venv (portable), fall back to the original machine's venv
set "PY=%~dp0.venv\Scripts\python.exe"
rem fallback: the "0 - setup" script creates .venv in this folder

if not exist "%PY%" (
    echo.
    echo [ERROR] Python environment not found.
    echo         Double-click the "0 - setup" script first.
    echo.
    pause
    exit /b 1
)

echo ================================================
echo   Record skeleton data
echo   1 = stand   2 = sit   3 = walk
echo   SPACE = start / stop recording
echo   s = screenshot     q = quit
echo ================================================
echo.
echo Make sure your whole body is in frame (knees visible).
echo.

"%PY%" recorder.py

echo.
echo Done. Data saved to: %~dp0data\raw
pause
