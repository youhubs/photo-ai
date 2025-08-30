# Makefile for Photo AI project
# Provides convenient commands for development, testing, and building

.PHONY: help install install-dev install-build clean test lint format build-pyinstaller package-windows package-macos package-linux run-gui run-cli

# Default target
help:
	@echo "Photo AI - Makefile Commands"
	@echo "=============================="
	@echo ""
	@echo "Virtual Environment:"
	@echo "  venv            Create virtual environment (.venv)"
	@echo "  venv-activate   Show activation commands"
	@echo "  venv-check      Check if in virtual environment"
	@echo "  setup-full      Create venv + install dev dependencies"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  setup           Quick setup (existing venv + install deps)"
	@echo "  setup-full      Complete setup (create venv + install deps)"
	@echo "  install         Install the package in development mode"
	@echo "  install-dev     Install with development dependencies"
	@echo "  install-build   Install with build dependencies"
	@echo "  clean          Clean build artifacts and cache files"
	@echo ""
	@echo "Quality & Testing:"
	@echo "  test            Run test suite"
	@echo "  lint            Run linting checks"
	@echo "  format          Format code with black"
	@echo ""
	@echo "Building:"
	@echo "  build-pyinstaller    Build with PyInstaller"
	@echo ""
	@echo "Packaging:"
	@echo "  package-windows      Create Windows installer"
	@echo "  package-macos        Create macOS app bundle/DMG"
	@echo "  package-linux        Create Linux packages"
	@echo ""
	@echo "Running:"
	@echo "  run-gui         Run the desktop GUI application"
	@echo "  run-cli         Run the CLI version"

# Virtual Environment Management
venv:
	@echo "🔧 Creating virtual environment..."
	python -m venv .venv
	@echo "✅ Virtual environment created in .venv/"
	@echo "📌 Activate with:"
	@echo "   Source: source .venv/bin/activate"
	@echo "   Windows: .venv\\Scripts\\activate"

venv-activate:
	@echo "📌 To activate virtual environment:"
	@echo "   macOS/Linux: source .venv/bin/activate"
	@echo "   Windows: .venv\\Scripts\\activate"

venv-check:
	@python -c "import sys; print('✅ Using virtual env' if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) else '❌ Not in virtual env')"

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-build:
	pip install -e ".[build]"

install-all:
	pip install -e ".[all]"

install-requirements:
	pip install -r requirements.txt

install-requirements-all:
	pip install -r requirements.txt

# Cleanup targets
clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "✅ Cleanup completed"

# Quality & testing targets
test:
	python -m pytest tests/ -v

test-coverage:
	python -m pytest tests/ --cov=photo_ai --cov-report=html --cov-report=term

lint:
	python -m flake8 photo_ai/
	python -m mypy photo_ai/

format:
	python -m black photo_ai/ tests/

format-check:
	python -m black --check photo_ai/ tests/

# Build targets
build-pyinstaller:
	python scripts/build.py pyinstaller --clean

build-pyinstaller-onefile:
	python scripts/build.py pyinstaller --clean --onefile


# Platform-specific packaging
package-windows:
	python scripts/build.py pyinstaller --clean --installer=windows

package-macos:
	python scripts/build.py pyinstaller --clean --installer=macos
	@if [ -d "dist/Photo AI.app" ]; then \
		echo "Creating DMG..."; \
		hdiutil create -srcfolder dist/ -volname "Photo AI" PhotoAI.dmg; \
		echo "✅ DMG created: PhotoAI.dmg"; \
	fi

package-linux:
	python scripts/build.py pyinstaller --clean --installer=linux

# Development run targets
run-gui:
	python photo_ai_gui.py

run-cli:
	python -m photo_ai.cli.main

# Docker targets (future)
docker-build:
	@echo "🐳 Docker build not implemented yet"

docker-run:
	@echo "🐳 Docker run not implemented yet"

# Release targets
pre-commit: format lint test
	@echo "✅ Pre-commit checks passed"

release-check: clean install-all pre-commit build-pyinstaller
	@echo "🚀 Release candidate ready"

# Platform detection
detect-platform:
	@python -c "import platform; print(f'Platform: {platform.system()} {platform.machine()}')"

# Quick development setup
setup: venv-check install-dev
	@echo "🛠️ Development environment set up"
	@echo "Run 'make run-gui' to start the application"

setup-full: venv install-dev
	@echo "🚀 Complete development setup with virtual environment"
	@echo "📌 Don't forget to activate: source .venv/bin/activate"

# CI/CD helpers
ci-install:
	pip install -e ".[all]"

ci-test: test lint

ci-build: build-pyinstaller

# Help for specific targets
help-build:
	@echo "Build System Help"
	@echo "=================="
	@echo ""
	@echo "PyInstaller:"
	@echo "  - Creates standalone executables"
	@echo "  - Fast startup, larger file size"
	@echo "  - Good for distribution"
	@echo ""
	@echo "Briefcase:"
	@echo "  - Creates native app packages"
	@echo "  - Platform-specific installers"
	@echo "  - Better OS integration"
	@echo ""
	@echo "Usage:"
	@echo "  make build-pyinstaller    # Quick executable"
	@echo "  make build-briefcase      # Native package"
	@echo "  make build-both           # Both approaches"

# Version management
version:
	@python -c "from photo_ai import __version__; print(__version__)"

bump-version:
	@echo "Version bumping not implemented yet"
	@echo "Edit photo_ai/__init__.py and pyproject.toml manually"