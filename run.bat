@echo off
title Deepfake Detection System
cd /d "D:\college\Sem 3\Macro Project-I\project df"
echo ========================================================
echo   Deepfake Detection & Forensic Analysis System
echo   Hardware: NVIDIA GeForce RTX 2050 (ONNX Accelerated)
echo ========================================================
echo Starting application...
python launcher.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo An error occurred while launching. Press any key to exit.
    pause >nul
)
