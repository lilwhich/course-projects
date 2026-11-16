@echo off
chcp 65001 >nul
cd /d "%~dp0"
title classroom-cv edge

py -3.11 -c "import sys" >nul 2>&1
if not errorlevel 1 goto USE311
py -3 -c "import sys" >nul 2>&1
if not errorlevel 1 goto USE3
python -c "import sys" >nul 2>&1
if not errorlevel 1 goto USEPY
goto NOPY

:USE311
set "PY=py -3.11"
goto CHECK
:USE3
set "PY=py -3"
goto CHECK
:USEPY
set "PY=python"
goto CHECK

:CHECK
echo ============================================================
echo   classroom-cv  edge camera
echo   python : %PY%
echo ============================================================
%PY% -c "import cv2,fastapi,uvicorn,requests" >nul 2>&1
if errorlevel 1 goto INSTALL
echo [1/2] dependencies ready
goto RUN

:INSTALL
echo [1/2] installing dependencies, about 1 minute ...
%PY% -m pip install --proxy "" -i https://mirrors.aliyun.com/pypi/simple/ --timeout 60 opencv-python==4.11.0.86 fastapi uvicorn requests
if errorlevel 1 goto PIPFAIL
echo       done

:RUN
echo [2/2] starting camera service. KEEP THIS WINDOW OPEN.
echo.
%PY% -m edge.camera_server --role out %*
pause
exit /b 0

:NOPY
echo.
echo [ERROR] Python not found on this machine.
echo.
echo   1. Download Python 3.11 from https://www.python.org/downloads/
echo   2. During install, CHECK the box "Add python.exe to PATH"
echo   3. Reopen this window and double click this file again
echo.
pause
exit /b 1

:PIPFAIL
echo.
echo [ERROR] pip install failed. Network problem most likely.
echo   Try again, or connect both PCs to the same phone hotspot.
echo.
pause
exit /b 1
