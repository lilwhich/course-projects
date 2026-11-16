@echo off
chcp 65001 >nul
cd /d "%~dp0"
title classroom-cv launcher

py -3.11 -c "import sys" >nul 2>&1
if not errorlevel 1 goto USE311
py -3 -c "import sys" >nul 2>&1
if not errorlevel 1 goto USE3
python -c "import sys" >nul 2>&1
if not errorlevel 1 goto USEPY
echo [ERROR] Python not found.
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
echo starting master + local camera ...
start "ccv master" cmd /k "%PY% -m master.app"
start "ccv local cam" cmd /k "%PY% -m edge.camera_server --port 8767 --role in"
timeout /t 5 >nul
start "" "http://127.0.0.1:8765/both"
timeout /t 2 >nul
start "" "http://127.0.0.1:8765/board"
echo.
echo   board : http://127.0.0.1:8765/board
echo   both  : http://127.0.0.1:8765/both
echo   grid  : http://127.0.0.1:8765/cameras
echo.
timeout /t 3 >nul
