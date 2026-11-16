@echo off
chcp 65001 >nul
cd /d "%~dp0"
title classroom-cv environment check

echo ============================================================
echo   environment check
echo ============================================================
echo.
echo [1] python launcher
where py 2>nul
if errorlevel 1 echo     py      NOT FOUND
where python 2>nul
if errorlevel 1 echo     python  NOT FOUND
echo.
echo [2] python version
py -3.11 --version 2>nul
py -3    --version 2>nul
python   --version 2>nul
echo.
echo [3] packages
py -3.11 -c "import cv2,fastapi,uvicorn,requests;print('     all OK')" 2>nul
python  -c "import cv2,fastapi,uvicorn,requests;print('     all OK')" 2>nul
echo.
echo [4] files
echo     dir  %CD%
if exist edge\camera_server.py (echo     edge   OK) else (echo     edge   MISSING)
if exist models\face_recognition_sface_2021dec.onnx (echo     models OK) else (echo     models MISSING)
echo.
pause
