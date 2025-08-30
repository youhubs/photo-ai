#!/bin/bash
# macOS/Linux shell script for building Photo AI
# Usage: ./build.sh [pyinstaller|briefcase|both] [options]

set -e  # Exit on any error

echo "🚀 Photo AI Unix Build Script"
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "❌ Python not found"
        echo "Please install Python 3.8+ and ensure it's in your PATH"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

echo "Using Python: $($PYTHON_CMD --version)"

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default arguments
BUILD_TOOL="${1:-pyinstaller}"
BUILD_ARGS="$BUILD_TOOL"

# Parse additional arguments
for arg in "$@"; do
    case $arg in
        --clean|--debug|--onefile|--dev)
            BUILD_ARGS="$BUILD_ARGS $arg"
            ;;
        --installer=*)
            BUILD_ARGS="$BUILD_ARGS $arg"
            ;;
    esac
done

echo "Building with: $BUILD_ARGS"
echo "Project root: $PROJECT_ROOT"
echo

# Change to project directory
cd "$PROJECT_ROOT"

# Check if virtual environment should be used
if [[ -f "venv/bin/activate" ]]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
elif [[ -f ".venv/bin/activate" ]]; then
    echo "📦 Activating virtual environment..."
    source .venv/bin/activate
fi

# Install build dependencies if needed
echo "📦 Checking build dependencies..."

if ! $PYTHON_CMD -c "import pyinstaller" 2>/dev/null; then
    echo "Installing PyInstaller..."
    $PYTHON_CMD -m pip install pyinstaller
fi

if ! $PYTHON_CMD -c "import briefcase" 2>/dev/null; then
    echo "Installing Briefcase..."
    $PYTHON_CMD -m pip install briefcase
fi

# Platform-specific setup
case "$(uname -s)" in
    Darwin*)
        echo "🍎 macOS detected"
        PLATFORM="macos"
        # Check for Xcode command line tools
        if ! command -v gcc &> /dev/null; then
            echo "⚠️  Xcode command line tools might be needed:"
            echo "   xcode-select --install"
        fi
        ;;
    Linux*)
        echo "🐧 Linux detected"
        PLATFORM="linux"
        # Check for common build dependencies
        if ! pkg-config --exists gtk+-3.0 2>/dev/null; then
            echo "⚠️  GTK development libraries might be needed:"
            echo "   Ubuntu/Debian: sudo apt install libgtk-3-dev"
            echo "   Fedora: sudo dnf install gtk3-devel"
        fi
        ;;
    *)
        echo "❓ Unknown platform: $(uname -s)"
        PLATFORM="unknown"
        ;;
esac

echo

# Run the build script
echo "🔨 Starting build process..."
$PYTHON_CMD scripts/build.py $BUILD_ARGS

if [[ $? -eq 0 ]]; then
    echo
    echo "🎉 Build completed successfully!"
    
    if [[ -d "dist" ]]; then
        echo "📁 Output files:"
        ls -la dist/
        
        # Platform-specific post-build actions
        case "$PLATFORM" in
            macos)
                if [[ -d "dist/Photo AI.app" ]]; then
                    echo
                    echo "🍎 macOS app bundle created!"
                    echo "To create DMG: hdiutil create -srcfolder dist/ -volname 'Photo AI' PhotoAI.dmg"
                fi
                ;;
            linux)
                echo
                echo "🐧 Linux build completed!"
                echo "Consider creating AppImage or system packages"
                ;;
        esac
    fi
else
    echo
    echo "💥 Build failed!"
    exit 1
fi