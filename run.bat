@echo off
title AI Traffic Analyzer - Startup Script
echo ====================================================
echo      AI-Based Intelligent Traffic Analyzer
echo            Startup Script (Windows)
echo ====================================================
echo.

:: Step 1: Ensure dependencies are installed
echo [1/3] Verifying and installing web dependencies...
pip install fastapi uvicorn python-multipart requests --quiet
if %errorlevel% neq 0 (
    echo [WARNING] Failed to verify/install pip dependencies. Attempting to proceed...
) else (
    echo [SUCCESS] Dependencies verified.
)
echo.

:: Step 2: Launch the web browser
echo [2/3] Preparing web browser...
timeout /t 2 /nobreak >nul
start http://localhost:8000
echo [SUCCESS] Browser launched for http://localhost:8000.
echo.

:: Step 3: Start the FastAPI backend server
echo [3/3] Starting the FastAPI server on port 8000...
echo Press Ctrl+C in this window to stop the server at any time.
echo.
python app.py

pause
