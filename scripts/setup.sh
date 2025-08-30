#!/bin/bash
# macOS/Linux shell script for Photo AI development setup
# This script creates a virtual environment and installs dev dependencies

set -e  # Exit on any error

echo "🚀 Photo AI Setup (Unix)"
echo "====================================="
echo

# Check if Python is available
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Python not found"
    echo "Please install Python 3.8+ and ensure it's in your PATH"
    echo "macOS: brew install python"
    echo "Ubuntu/Debian: sudo apt install python3 python3-venv python3-pip"
    echo "Fedora: sudo dnf install python3 python3-venv python3-pip"
    exit 1
fi

echo "🐍 Python version:"
$PYTHON_CMD --version
echo

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PATH="$PROJECT_ROOT/.venv"

echo "📁 Project root: $PROJECT_ROOT"
echo

# Change to project directory
cd "$PROJECT_ROOT"

# Create virtual environment if it doesn't exist
if [[ -d "$VENV_PATH" ]]; then
    echo "📦 Virtual environment already exists"
else
    echo "🔧 Creating virtual environment..."
    $PYTHON_CMD -m venv "$VENV_PATH"
    echo "✅ Virtual environment created"
fi

echo

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Verify activation
if [[ "$VIRTUAL_ENV" == "$VENV_PATH" ]]; then
    echo "✅ Virtual environment activated: $VIRTUAL_ENV"
else
    echo "⚠️  Virtual environment activation unclear"
fi

# Platform-specific system dependencies check
case "$(uname -s)" in
    Darwin*)
        echo "🍎 macOS detected"
        # Check for Xcode command line tools
        if ! command -v gcc &> /dev/null; then
            echo "⚠️  Xcode command line tools recommended:"
            echo "   xcode-select --install"
        fi
        ;;
    Linux*)
        echo "🐧 Linux detected"
        # Check for common build dependencies
        if ! pkg-config --exists gtk+-3.0 2>/dev/null; then
            echo "⚠️  GTK development libraries recommended:"
            echo "   Ubuntu/Debian: sudo apt install libgtk-3-dev libgl1-mesa-dev"
            echo "   Fedora: sudo dnf install gtk3-devel mesa-libGL-devel"
        fi
        ;;
esac

echo

# Upgrade pip
echo "📦 Upgrading pip..."
$PYTHON_CMD -m pip install --upgrade pip

# Install development dependencies
echo "📦 Installing development dependencies..."
pip install -e ".[dev]"

echo
echo "✅ Installation completed!"

# Run basic tests
echo
echo "🧪 Testing basic imports..."
$PYTHON_CMD -c "import photo_ai; print('✅ Photo AI core imported')" || echo "❌ Core import failed"
$PYTHON_CMD -c "from photo_ai.gui.app import PhotoAIApp; print('✅ GUI components imported')" || echo "⚠️  GUI import failed (PyQt6 issue?)"

# Create .vscode settings if VS Code is detected
if command -v code &> /dev/null; then
    echo
    echo "💻 VS Code detected - creating workspace settings..."
    mkdir -p .vscode
    cat > .vscode/settings.json << EOF
{
    "python.interpreterPath": "./.venv/bin/python",
    "python.terminal.activateEnvironment": true,
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".venv": true,
        "build": true,
        "dist": true
    }
}
EOF
    echo "✅ VS Code settings created"
fi

echo
echo "🎉 Development environment setup complete!"
echo
echo "📋 Next steps:"
echo "=============="
echo "1. Activate virtual environment:"
echo "   source .venv/bin/activate"
echo
echo "2. Run the application:"
echo "   python photo_ai_gui.py          # GUI version"
echo "   python -m photo_ai.cli.main     # CLI version"
echo
echo "3. Run tests:"
echo "   pytest tests/"
echo
echo "4. Build application:"
echo "   make build-pyinstaller"
echo "   # or: python scripts/build.py pyinstaller"
echo
echo "💡 Useful commands:"
echo "   make help           # Show all available commands"
echo "   make test          # Run test suite"
echo "   make lint          # Check code quality"
echo "   make format        # Format code"
echo "   make run-gui       # Launch GUI application"
echo

# Check if we should stay in the virtual environment
echo "🔧 Virtual environment is ready!"
echo "Run 'source .venv/bin/activate' to activate it manually"