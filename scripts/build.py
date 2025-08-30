#!/usr/bin/env python3
"""
Build script for Photo AI desktop application.
Builds using PyInstaller for cross-platform distribution.
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path


class PhotoAIBuilder:
    """Builder class for Photo AI desktop application."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.build_dir = project_root / "build"
        self.dist_dir = project_root / "dist"

    def clean(self):
        """Clean build artifacts."""
        print("🧹 Cleaning build artifacts...")

        dirs_to_clean = [
            self.build_dir,
            self.dist_dir,
            self.project_root / "*.egg-info",
            self.project_root / "__pycache__",
        ]

        for dir_path in dirs_to_clean:
            if dir_path.exists():
                if dir_path.is_dir():
                    shutil.rmtree(dir_path)
                    print(f"   Removed directory: {dir_path}")
                else:
                    dir_path.unlink()
                    print(f"   Removed file: {dir_path}")

        # Clean Python cache files recursively
        for root, dirs, files in os.walk(self.project_root):
            for dir_name in dirs[:]:  # Create copy to modify during iteration
                if dir_name == "__pycache__":
                    shutil.rmtree(Path(root) / dir_name)
                    dirs.remove(dir_name)

            for file_name in files:
                if file_name.endswith((".pyc", ".pyo")):
                    (Path(root) / file_name).unlink()

        print("✅ Cleanup completed")

    def check_dependencies(self):
        """Check if required build dependencies are available."""
        print(f"🔍 Checking PyInstaller dependencies...")

        try:
            import PyInstaller

            print(f"   ✅ PyInstaller found")
            return True
        except ImportError:
            print(f"   ❌ PyInstaller missing")
            print(f"\n📦 Install missing dependencies:")
            print(f"   pip install pyinstaller")
            return False

    def build_with_pyinstaller(self, options: dict):
        """Build using PyInstaller."""
        print("🔨 Building with PyInstaller...")

        spec_file = self.project_root / "photo-ai.spec"
        if not spec_file.exists():
            print("❌ PyInstaller spec file not found: photo-ai.spec")
            return False

        cmd = [sys.executable, "-m", "PyInstaller", "--clean", str(spec_file)]

        if options.get("debug"):
            cmd.append("--debug=all")

        if options.get("onefile"):
            cmd.append("--onefile")

        print(f"   Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, cwd=self.project_root, check=True)
            print("✅ PyInstaller build completed successfully")

            # Show output files
            dist_contents = list(self.dist_dir.iterdir()) if self.dist_dir.exists() else []
            if dist_contents:
                print("\n📦 Built files:")
                for item in dist_contents:
                    size = self._get_size_str(item)
                    print(f"   {item.name} ({size})")

            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ PyInstaller build failed with return code {e.returncode}")
            return False

    def create_installer(self, platform: str):
        """Create platform-specific installer."""
        print(f"📦 Creating installer for {platform}...")

        if platform == "windows":
            self._create_windows_installer()
        elif platform == "macos":
            self._create_macos_installer()
        elif platform == "linux":
            self._create_linux_installer()
        else:
            print(f"❌ Unsupported platform: {platform}")
            return False

        return True

    def _create_windows_installer(self):
        """Create Windows installer using NSIS or similar."""
        print("   Windows installer creation not implemented yet")
        print("   Consider using: Inno Setup, NSIS, or WiX")

    def _create_macos_installer(self):
        """Create macOS installer."""
        app_path = self.dist_dir / "Photo AI.app"
        if app_path.exists():
            print(f"   macOS app bundle created: {app_path}")
            print("   To create DMG: hdiutil create -srcfolder dist/ PhotoAI.dmg")
        else:
            print("   ❌ macOS app bundle not found")

    def _create_linux_installer(self):
        """Create Linux installer (AppImage, deb, rpm)."""
        print("   Linux installer creation not implemented yet")
        print("   Consider using: AppImage, dpkg, or rpm")

    def _get_size_str(self, path: Path) -> str:
        """Get human-readable size string for a file or directory."""
        if path.is_file():
            size = path.stat().st_size
        elif path.is_dir():
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        else:
            return "unknown"

        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"


def main():
    """Main build script entry point."""
    parser = argparse.ArgumentParser(description="Build Photo AI desktop application")

    parser.add_argument("tool", choices=["pyinstaller"], help="Build tool to use")

    parser.add_argument(
        "--clean", action="store_true", help="Clean build artifacts before building"
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug mode for builds")

    parser.add_argument(
        "--onefile", action="store_true", help="Create single executable file (PyInstaller only)"
    )

    parser.add_argument(
        "--installer",
        choices=["windows", "macos", "linux"],
        help="Create platform-specific installer",
    )

    args = parser.parse_args()

    # Initialize builder
    project_root = Path(__file__).parent.parent
    builder = PhotoAIBuilder(project_root)

    print("🚀 Photo AI Build Script")
    print(f"   Project: {project_root}")
    print(f"   Tool: {args.tool}")

    # Clean if requested
    if args.clean:
        builder.clean()

    # Check dependencies
    if not builder.check_dependencies():
        sys.exit(1)

    # Build options
    options = {
        "debug": args.debug,
        "onefile": args.onefile,
    }

    # Execute build
    success = builder.build_with_pyinstaller(options)

    # Create installer if requested
    if args.installer and success:
        success &= builder.create_installer(args.installer)

    if success:
        print("\n🎉 Build completed successfully!")
        if builder.dist_dir.exists():
            print(f"📁 Output directory: {builder.dist_dir}")
    else:
        print("\n💥 Build failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
