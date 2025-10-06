#!/bin/bash
# MAMBA_SLM Dashboard Launcher for Linux/Mac
# Run: chmod +x launch_dashboard.sh && ./launch_dashboard.sh

echo ""
echo "========================================"
echo "  MAMBA_SLM Unified Dashboard Launcher"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ ERROR: Python 3 is not installed"
    echo "Please install Python 3.8+ from python.org"
    exit 1
fi

echo "Checking Python version..."
python3 --version

echo ""
echo "Checking dependencies..."

# Check for PyQt6
if ! python3 -c "import PyQt6" 2>/dev/null; then
    echo ""
    echo "⚠️  WARNING: PyQt6 not found!"
    echo "Installing required dependencies..."
    echo ""
    python3 -m pip install PyQt6
    if [ $? -ne 0 ]; then
        echo "❌ ERROR: Failed to install PyQt6"
        exit 1
    fi
fi

echo "✓ Dependencies OK"
echo ""
echo "Launching dashboard..."
echo ""

# Launch the dashboard
python3 launch_dashboard.py

# Check exit status
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Dashboard exited with error"
    exit 1
fi
