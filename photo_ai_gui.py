#!/usr/bin/env python3
"""
Photo AI Desktop Application

A standalone entry point for the Photo AI GUI application.
This file can be run directly without installing the package.
"""

import sys
import os

# Add the package directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from photo_ai.gui.app import main
except ImportError as e:
    print(f"Error importing Photo AI GUI: {e}")
    print("Please ensure PyQt6 is installed:")
    print("  pip install PyQt6")
    print("Or install the complete package:")
    print("  pip install -e .")
    sys.exit(1)

if __name__ == '__main__':
    sys.exit(main())