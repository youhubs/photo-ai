#!/usr/bin/env python3
"""
Development environment setup script for Photo AI.
This script helps set up a complete development environment.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


class EnvironmentSetup:
    """Development environment setup helper."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.venv_path = self.project_root / ".venv"
        self.platform = platform.system().lower()

    def print_header(self):
        """Print setup header."""
        print("🚀 Photo AI Environment Setup")
        print("=" * 32)
        print(f"📁 Project: {self.project_root}")
        print(f"🖥️  Platform: {platform.system()} {platform.machine()}")
        print(f"🐍 Python: {sys.version.split()[0]}")
        print()

    def check_python_version(self):
        """Check if Python version is compatible."""
        version = sys.version_info
        if version < (3, 8):
            print("❌ Python 3.8+ required!")
            print(f"   Current version: {version.major}.{version.minor}")
            print("   Please upgrade Python and try again.")
            return False

        print(f"✅ Python {version.major}.{version.minor} is compatible")
        return True

    def create_virtual_environment(self):
        """Create virtual environment if it doesn't exist."""
        if self.venv_path.exists():
            print(f"📦 Virtual environment already exists: {self.venv_path}")
            return True

        print("🔧 Creating virtual environment...")
        try:
            subprocess.run(
                [sys.executable, "-m", "venv", str(self.venv_path)],
                check=True,
                cwd=self.project_root,
            )

            print(f"✅ Virtual environment created: {self.venv_path}")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create virtual environment: {e}")
            return False

    def get_activation_command(self):
        """Get the appropriate activation command for the platform."""
        if self.platform == "windows":
            return str(self.venv_path / "Scripts" / "activate.bat")
        else:
            return f"source {self.venv_path}/bin/activate"

    def get_python_executable(self):
        """Get the virtual environment Python executable."""
        if self.platform == "windows":
            return str(self.venv_path / "Scripts" / "python.exe")
        else:
            return str(self.venv_path / "bin" / "python")

    def install_dependencies(self):
        """Install development dependencies in virtual environment."""
        python_exe = self.get_python_executable()

        if not Path(python_exe).exists():
            print("❌ Virtual environment Python not found")
            return False

        print("📦 Installing development dependencies...")

        try:
            # Upgrade pip first
            subprocess.run(
                [python_exe, "-m", "pip", "install", "--upgrade", "pip"],
                check=True,
                cwd=self.project_root,
            )

            # Install package in development mode with dev dependencies
            subprocess.run(
                [python_exe, "-m", "pip", "install", "-e", ".[dev]"],
                check=True,
                cwd=self.project_root,
            )

            print("✅ Dependencies installed successfully")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            print("\n💡 You can try installing manually:")
            print(f"   {self.get_activation_command()}")
            print(f'   pip install -e ".[dev]"')
            return False

    def run_basic_tests(self):
        """Run basic tests to verify installation."""
        python_exe = self.get_python_executable()

        print("🧪 Running basic tests...")

        # Test imports
        test_imports = [
            "import photo_ai",
            "from photo_ai.core.config import Config",
            "from photo_ai.core.photo_processor import PhotoProcessor",
        ]

        for test_import in test_imports:
            try:
                subprocess.run([python_exe, "-c", test_import], check=True, capture_output=True)
                print(f"   ✅ {test_import}")
            except subprocess.CalledProcessError:
                print(f"   ❌ {test_import}")
                return False

        # Test GUI imports (optional)
        try:
            subprocess.run(
                [python_exe, "-c", "from photo_ai.gui.app import PhotoAIApp"],
                check=True,
                capture_output=True,
            )
            print("   ✅ GUI imports successful")
        except subprocess.CalledProcessError:
            print("   ⚠️  GUI imports failed (PyQt6 might not be available)")

        print("✅ Basic tests passed")
        return True

    def create_vscode_config(self):
        """Create VS Code configuration for the project."""
        vscode_dir = self.project_root / ".vscode"
        vscode_dir.mkdir(exist_ok=True)

        # Python interpreter setting
        settings_json = vscode_dir / "settings.json"
        python_path = self.get_python_executable()

        settings_content = f"""{{
    "python.interpreterPath": "{python_path.replace(os.sep, '/')}",
    "python.terminal.activateEnvironment": true,
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "files.exclude": {{
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".venv": true,
        "build": true,
        "dist": true
    }}
}}"""

        try:
            with open(settings_json, "w") as f:
                f.write(settings_content)
            print(f"✅ VS Code settings created: {settings_json}")
        except Exception as e:
            print(f"⚠️  Could not create VS Code settings: {e}")

    def print_next_steps(self):
        """Print next steps for the user."""
        activation_cmd = self.get_activation_command()

        print("\n🎉 Development environment setup complete!")
        print("\n📋 Next steps:")
        print("=" * 20)
        print(f"1. Activate virtual environment:")
        print(f"   {activation_cmd}")
        print()
        print("2. Verify installation:")
        print("   python -c \"import photo_ai; print('Photo AI imported successfully!')\"")
        print()
        print("3. Run the application:")
        print("   python photo_ai_gui.py          # GUI version")
        print("   python -m photo_ai.cli.main     # CLI version")
        print()
        print("4. Run tests:")
        print("   pytest tests/")
        print()
        print("5. Build application:")
        print("   make build-pyinstaller")
        print()
        print("💡 Useful commands:")
        print("   make help           # Show all available commands")
        print("   make test          # Run test suite")
        print("   make lint          # Check code quality")
        print("   make format        # Format code")

    def run_setup(self, skip_tests=False):
        """Run the complete setup process."""
        self.print_header()

        if not self.check_python_version():
            return False

        if not self.create_virtual_environment():
            return False

        if not self.install_dependencies():
            return False

        if not skip_tests and not self.run_basic_tests():
            print("⚠️  Setup completed but tests failed")

        self.create_vscode_config()
        self.print_next_steps()
        return True


def main():
    """Main setup script."""
    import argparse

    parser = argparse.ArgumentParser(description="Set up Photo AI development environment")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running basic tests")
    parser.add_argument("--no-vscode", action="store_true", help="Skip VS Code configuration")

    args = parser.parse_args()

    try:
        setup = EnvironmentSetup()
        success = setup.run_setup(skip_tests=args.skip_tests)

        if success:
            print("\n✅ Setup completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Setup failed!")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⚠️  Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
