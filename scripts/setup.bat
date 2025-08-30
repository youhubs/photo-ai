@echo off
REM Windows batch script for Photo AI development setup
REM This script creates a virtual environment and installs dev dependencies

setlocal enabledelayedexpansion

echo 🚀 Photo AI Setup (Windows)
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found in PATH
    echo Please install Python 3.8+ and add it to PATH
    echo Download from: https://python.org/downloads/
    pause
    exit /b 1
)

echo 🐍 Python version:
python --version
echo.

REM Get script directory and project root
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
set VENV_PATH=%PROJECT_ROOT%\.venv

echo 📁 Project root: %PROJECT_ROOT%
echo.

REM Create virtual environment if it doesn't exist
if exist "%VENV_PATH%" (
    echo 📦 Virtual environment already exists
) else (
    echo 🔧 Creating virtual environment...
    python -m venv "%VENV_PATH%"
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
    echo ✅ Virtual environment created
)

echo.

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call "%VENV_PATH%\Scripts\activate.bat"

REM Upgrade pip
echo 📦 Upgrading pip...
python -m pip install --upgrade pip

REM Install development dependencies
echo 📦 Installing development dependencies...
pip install -e ".[dev]"

if errorlevel 1 (
    echo ❌ Failed to install dependencies
    echo.
    echo 💡 You can try manually:
    echo    .venv\Scripts\activate
    echo    pip install -e ".[dev]"
    pause
    exit /b 1
)

echo.
echo ✅ Installation completed!

REM Run basic tests
echo 🧪 Testing basic imports...
python -c "import photo_ai; print('✅ Photo AI core imported')" || echo "❌ Core import failed"
python -c "from photo_ai.gui.app import PhotoAIApp; print('✅ GUI components imported')" || echo "⚠️ GUI import failed (PyQt6 issue?)"

echo.
echo 🎉 Development environment setup complete!
echo.
echo 📋 Next steps:
echo ============
echo 1. Activate virtual environment:
echo    .venv\Scripts\activate.bat
echo.
echo 2. Run the application:
echo    python photo_ai_gui.py
echo.
echo 3. Run tests:
echo    pytest tests\
echo.
echo 4. Build application:
echo    python scripts\build.py pyinstaller
echo.
echo 💡 Useful commands:
echo    make help          # Show all commands (requires make)
echo    python scripts\build.py --help    # Build options
echo.

pause