@echo off
REM MAMBA_SLM Dashboard Launcher for Windows
REM Double-click this file to launch the dashboard

echo.
echo ========================================
echo   MAMBA_SLM Unified Dashboard Launcher
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

echo Checking Python version...
python --version

echo.
echo Checking dependencies...
python -c "import PyQt6" >nul 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: PyQt6 not found!
    echo Installing required dependencies...
    echo.
    python -m pip install PyQt6
    if errorlevel 1 (
        echo ERROR: Failed to install PyQt6
        pause
        exit /b 1
    )
)

echo Dependencies OK
echo.
echo Launching dashboard...
echo.

REM Launch the dashboard
python launch_dashboard.py

REM If dashboard exits with error, pause to see error message
if errorlevel 1 (
    echo.
    echo Dashboard exited with error
    pause
)
