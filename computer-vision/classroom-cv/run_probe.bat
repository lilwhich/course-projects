@echo off
chcp 65001 >nul
cd /d "%~dp0"

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
%PY% -m edge.probe %*
pause
