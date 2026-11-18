@echo off
chcp 65001 >nul
title PoseAction 7 - Phone Camera
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
echo   Phone camera (wireless)
echo.
echo   Turns your phone into the camera for this app.
echo   Useful because a laptop camera is usually too
echo   high to see your legs - and standing vs sitting
echo   cannot be told apart without seeing the legs.
echo.
echo   The wizard will scan your Wi-Fi for the phone,
echo   or you can type the address shown in the phone
echo   app, e.g. 192.168.1.7:8080
echo ================================================
echo.

"%PY%" setup_phone_cam.py %*

echo.
pause
