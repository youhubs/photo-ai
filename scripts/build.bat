@echo off
REM Windows batch script for building Photo AI
REM Usage: build.bat [pyinstaller|briefcase|both] [options]

setlocal enabledelayedexpansion

echo 🚀 Photo AI Windows Build Script
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found in PATH
    echo Please install Python 3.8+ and add it to PATH
    exit /b 1
)

REM Get the script directory
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..

REM Default arguments
set BUILD_TOOL=%1
if "%BUILD_TOOL%"=="" set BUILD_TOOL=pyinstaller

REM Build arguments
set BUILD_ARGS=%BUILD_TOOL%
if "%2"=="--clean" set BUILD_ARGS=%BUILD_ARGS% --clean
if "%3"=="--debug" set BUILD_ARGS=%BUILD_ARGS% --debug
if "%2"=="--onefile" set BUILD_ARGS=%BUILD_ARGS% --onefile
if "%3"=="--onefile" set BUILD_ARGS=%BUILD_ARGS% --onefile

echo Building with: %BUILD_ARGS%
echo Project root: %PROJECT_ROOT%
echo.

REM Change to project directory
cd /d "%PROJECT_ROOT%"

REM Install build dependencies if needed
echo 📦 Checking build dependencies...
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

pip show briefcase >nul 2>&1
if errorlevel 1 (
    echo Installing Briefcase...
    pip install briefcase
)

REM Run the build script
echo.
echo 🔨 Starting build process...
python scripts/build.py %BUILD_ARGS%

if errorlevel 1 (
    echo.
    echo 💥 Build failed!
    exit /b 1
)

echo.
echo 🎉 Build completed successfully!
echo 📁 Check the dist/ directory for output files

REM Optional: Open dist folder
if exist "dist\" (
    set /p OPEN_FOLDER="Open dist folder? (y/N): "
    if /i "!OPEN_FOLDER!"=="y" explorer dist
)

endlocal