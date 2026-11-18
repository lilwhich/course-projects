@echo off
chcp 65001 >nul
title PoseAction 6 - Export Excel
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
echo   Merge all CSV clips into one Excel file
echo   Output: reports\dataset.xlsx
echo   One sheet per clip + a summary sheet
echo ================================================
echo.

"%PY%" to_excel.py

echo.
echo Opening reports folder...
start "" "%~dp0reports"
pause
