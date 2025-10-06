#!/usr/bin/env python3
"""
Quick launcher script for MAMBA_SLM Dashboard
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_dependencies():
    """Check if required dependencies are installed"""
    missing = []
    
    try:
        import PyQt6
    except ImportError:
        missing.append("PyQt6")
    
    try:
        import torch
    except ImportError:
        missing.append("torch")
    
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
    
    if missing:
        print("❌ Missing required dependencies:")
        for dep in missing:
            print(f"   - {dep}")
        print("\nPlease install dependencies:")
        print("   pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Launch the dashboard"""
    print("🐍 MAMBA_SLM Unified Dashboard Launcher")
    print("=" * 50)
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    print("✓ All dependencies installed")
    print("\nLaunching dashboard...")
    
    try:
        from mamba_dashboard import main as dashboard_main
        dashboard_main()
    except Exception as e:
        print(f"\n❌ Error launching dashboard:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
