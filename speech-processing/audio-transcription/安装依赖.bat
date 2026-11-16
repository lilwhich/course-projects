@echo off
chcp 936 >nul
title 安装依赖 - 录音转文字
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

echo.
echo 使用的 Python：%PY%
%PY% --version
echo.
echo 正在安装 faster-whisper 与 av（清华源）...
echo.
%PY% -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
echo.
echo 安装结束。首次运行程序时会自动下载模型（约 461 MB）。
pause
