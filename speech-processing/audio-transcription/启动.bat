@echo off
chcp 936 >nul
title 录音转文字
cd /d "%~dp0"

set "PY="
py -3.11 -c "import sys" >nul 2>nul && set "PY=py -3.11"
if not defined PY py -3 -c "import sys" >nul 2>nul && set "PY=py -3"
if not defined PY python -c "import sys" >nul 2>nul && set "PY=python"
if not defined PY (
    echo [错误] 没有找到 Python，请先安装 Python 3.9 以上版本并勾选 Add to PATH。
    pause
    exit /b 1
)

if "%~1"=="" goto ask

echo.
echo ===== 开始转写 =====
echo.
%PY% "录音转文字.py" %*
echo.
pause
goto :eof

:ask
echo.
echo 把音频文件直接拖到本窗口里，然后按回车。
echo 也可以手敲路径。直接回车则退出。
echo.
set /p AUD="音频路径: "
if "%AUD%"=="" goto :eof
%PY% "录音转文字.py" "%AUD%"
echo.
pause
