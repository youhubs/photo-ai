"""Setup script for Photo AI package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="photo-ai",
    version="1.0.0",
    author="Photo AI Team",
    description="Advanced photo processing and analysis toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "transformers>=4.20.0",
        "opencv-python>=4.5.0",
        "face-recognition>=1.3.0",
        "Pillow>=9.0.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "PyQt6>=6.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.910",
        ],
        "build": [
            "briefcase>=0.3.15",
            "pyinstaller>=5.0",
            "auto-py-to-exe>=2.20.0",
        ],
        "all": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.910",
            "briefcase>=0.3.15",
            "pyinstaller>=5.0",
            "auto-py-to-exe>=2.20.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "photo-ai=photo_ai.cli.main:main",
        ],
        "gui_scripts": [
            "photo-ai-gui=photo_ai.gui.app:main",
        ],
    },
)
