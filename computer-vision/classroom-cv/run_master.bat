@echo off
chcp 65001 >nul
cd /d "%~dp0"
title classroom-cv master

py -3.11 -c "import sys" >nul 2>&1
if not errorlevel 1 goto USE311
py -3 -c "import sys" >nul 2>&1
if not errorlevel 1 goto USE3
python -c "import sys" >nul 2>&1
if not errorlevel 1 goto USEPY
echo [ERROR] Python not found. See setup_edge.bat for instructions.
pause
exit /b 1

:USE311
set "PY=py -3.11"
goto RUN
:USE3
set "PY=py -3"
goto RUN
:USEPY
set "PY=python"

:RUN
%PY% -c "import cv2,fastapi,uvicorn" >nul 2>&1
if errorlevel 1 goto NODEPS
%PY% -m master.app
pause
exit /b 0

:NODEPS
echo [ERROR] dependencies missing. Run once:
echo   %PY% -m pip install --proxy "" -i https://mirrors.aliyun.com/pypi/simple/ opencv-python==4.11.0.86 fastapi uvicorn requests
pause
exit /b 1
