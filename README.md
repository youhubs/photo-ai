# Photo AI 📸🤖

Advanced photo processing and analysis toolkit with both **desktop GUI** and **command-line interfaces**. Powered by machine learning to automatically organize, enhance, and process your photos with intelligent algorithms.

## Features

### 🔍 **Smart Photo Analysis**
- **Sharpness Detection**: Automatically identify blurry vs sharp photos using multiple algorithms
- **Duplicate Detection**: Find exact duplicates and similar photos using perceptual hashing and deep learning
- **Face Analysis**: Detect faces and validate photo quality for official documents
- **Quality Assessment**: Comprehensive photo quality scoring and analysis

### 📋 **Automated Organization**
- **Intelligent Sorting**: Automatically sort photos into good/bad quality folders
- **Best Photo Selection**: Select the best photos from similar groups
- **Time-based Clustering**: Group photos taken within time windows
- **Batch Processing**: Process thousands of photos efficiently

### 🎯 **Specialized Processing**
- **Visa Photo Creator**: Generate compliant visa/passport photos with precise dimensions
- **Background Removal**: Smart background removal and replacement
- **Face Detection**: Advanced face detection with quality validation
- **Custom Cropping**: Intelligent cropping based on face position and photo requirements

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd photo-ai

# Install the package (includes GUI support)
pip install -e .

# Alternative installation methods
pip install -r requirements.txt          # Core dependencies only
pip install -r requirements.txt           # All dependencies
```

### 🖥️ **Desktop Application (Recommended)**

Launch the user-friendly GUI application:

```bash
# Launch desktop GUI
photo-ai-gui

# Or run directly without installation
python photo_ai_gui.py
```

**GUI Features:**
- 📁 **Photo Selection**: Browse folders or select individual files with drag & drop
- 🖼️ **Built-in Viewer**: Photo viewer with thumbnails and navigation
- ⚡ **Real-time Progress**: Visual progress tracking with status updates
- 🎯 **One-click Processing**: Complete photo analysis and organization
- 📄 **Visa Photo Creator**: Specialized dialog with live preview and validation
- ⚙️ **Settings Panel**: Advanced configuration with tabbed interface
- 📊 **Results Dashboard**: Detailed analytics and processing history

### 💻 **Command Line Interface**

For automation and scripting:

```bash
# Process all photos in a directory
photo-ai process photos/

# Create a visa photo
photo-ai visa input.jpg visa_output.jpg

# Analyze photo quality without processing
photo-ai analyze photos/ --format text

# Show processing statistics
photo-ai stats
```

## Command Reference

### Photo Processing
```bash
# Full pipeline processing
photo-ai process [directory] [--good-dir DIR] [--bad-dir DIR]

# Quality analysis only
photo-ai analyze <directory> [--format json|text]

# Processing statistics
photo-ai stats
```

### Visa Photos
```bash
# Create compliant visa photo (33×48mm, 300 DPI)
photo-ai visa <input> [output] [--debug]
```

### Configuration
```bash
# Environment variables
export PHOTO_AI_INPUT_DIR="/path/to/input"
export PHOTO_AI_GOOD_DIR="/path/to/good"
export PHOTO_AI_BAD_DIR="/path/to/bad"
export PHOTO_AI_OUTPUT_DIR="/path/to/output"

# Command line options
photo-ai --input-dir photos/ --good-dir sharp --bad-dir blurry
```

## Python API

### Basic Usage
```python
from photo_ai import PhotoProcessor
from photo_ai.core.config import Config

# Create processor with default config
processor = PhotoProcessor()

# Process photos
result = processor.process_photos_pipeline("photos/")
print(f"Processed {result['total_images']} images")

# Create visa photo
visa_result = processor.process_visa_photo("input.jpg", "visa.jpg")
if visa_result['success']:
    print(f"Visa photo created: {visa_result['output_path']}")
```

### Advanced Configuration
```python
from photo_ai.core.config import Config

# Custom configuration
config = Config()
config.processing.sharpness_threshold = 0.8
config.processing.cluster_eps = 0.3
config.visa.photo_width_mm = 35  # Custom size

processor = PhotoProcessor(config)
```

### Individual Processors
```python
from photo_ai.processors.quality.sharpness import SharpnessAnalyzer
from photo_ai.processors.face.detector import FaceDetector

config = Config()

# Analyze sharpness
sharpness = SharpnessAnalyzer(config)
result = sharpness.analyze_comprehensive("photo.jpg")
print(f"Sharp: {result['overall_is_sharp']}")

# Detect faces
face_detector = FaceDetector(config)
faces = face_detector.detect_faces("portrait.jpg")
print(f"Found {faces['face_count']} faces")
```

## Technical Details

### Algorithms Used
- **Sharpness Detection**: Laplacian variance, gradient analysis, and ResNet-based ML models
- **Feature Extraction**: Vision Transformer (ViT) for deep image features
- **Clustering**: DBSCAN for grouping similar photos
- **Face Detection**: face_recognition library with dlib backend
- **Background Removal**: GrabCut algorithm with intelligent masking

### Requirements
- **Python**: 3.8+
- **Memory**: 4GB+ RAM recommended for large batches
- **GPU**: Optional CUDA support for faster processing
- **Storage**: Processed photos are copied, not moved (preserve originals)
- **Display**: For GUI - any modern desktop environment (Windows/macOS/Linux)

### Supported Formats
- **Input**: JPG, JPEG, PNG, WebP, BMP, TIFF
- **Output**: JPG (high quality), PNG for transparency
- **Metadata**: EXIF data preserved where possible

## Project Structure

```
photo_ai/
├── core/                   # Core processing logic
│   ├── config.py          # Configuration management
│   └── photo_processor.py # Main orchestrator
├── processors/            # Specialized processors
│   ├── quality/          # Quality assessment
│   ├── face/             # Face detection & processing
│   └── background/       # Background manipulation
├── utils/                # Utility functions
├── cli/                  # Command line interface
└── gui/                  # Desktop GUI application
    ├── app.py            # Main application with theming
    ├── main_window.py    # Primary interface window
    ├── widgets/          # Reusable GUI components
    └── dialogs/          # Modal dialogs (visa, settings)
```

## Troubleshooting

### Common Issues

**"No face detected"**
- Ensure photo shows a clear frontal face
- Face should occupy 25-45% of image height
- Use good lighting and avoid shadows

**"Image too small"**
- Minimum resolution: 1600x1200 pixels
- Higher resolution recommended for visa photos
- Resize image before processing

**"Model loading failed"**
- Check internet connection for initial model download
- Models are cached locally after first use
- ~2GB storage needed for all models

**GUI Not Starting**
- Ensure PyQt6 is installed: `pip install PyQt6`
- Check display environment variables on Linux
- Try running: `python photo_ai_gui.py` for direct execution

**Processing Stuck/Slow**
- Check system resources (CPU/Memory usage)
- Try smaller batch sizes in GUI settings
- Verify internet connection for initial model downloads

### Debug Mode
```bash
# Command line debug
photo-ai visa input.jpg output.jpg --debug
photo-ai process photos/ --verbose

# GUI debug
# Enable debug mode in visa photo dialog
# Check the Log tab for detailed processing information
```

## Building & Distribution

### 🔨 **Build Requirements**

```bash
# Install build dependencies
pip install -e ".[build]"

# Or install PyInstaller directly
pip install pyinstaller
```

### 🚀 **Quick Build**

**PyInstaller Build:**
```bash
# Cross-platform build script
python scripts/build.py pyinstaller --clean

# Platform-specific scripts
# Windows: scripts/build.bat pyinstaller
# macOS/Linux: scripts/build.sh pyinstaller

# Or use Makefile
make build-pyinstaller
```

### 📦 **Distribution Packages**

**Windows:**
```bash
# Create standalone executable
make package-windows

# Output: dist/PhotoAI.exe
```

**macOS:**
```bash
# Create app bundle and DMG
make package-macos

# Output: dist/Photo AI.app, PhotoAI.dmg
```

**Linux:**
```bash
# Create AppImage/packages  
make package-linux

# Output: dist/PhotoAI (executable)
```

### ⚙️ **Build Options**

```bash
# Single file executable (slower startup, smaller distribution)
python scripts/build.py pyinstaller --onefile

# Debug build (with console output)
python scripts/build.py pyinstaller --debug
```

### 🛠️ **Makefile Commands**

```bash
# Development
make install-dev      # Install with dev dependencies
make clean           # Clean build artifacts

# Building
make build-pyinstaller    # Build executable

# Quality checks
make test           # Run tests
make lint           # Check code quality
make format         # Format code

# Running
make run-gui        # Launch desktop app
make run-cli        # Launch CLI version
```

### 📋 **Platform-Specific Notes**

**Windows:**
- Requires Visual Studio Build Tools for some dependencies
- Windows Defender may flag executables (whitelist needed)
- Consider code signing for distribution

**macOS:**  
- Requires Xcode Command Line Tools: `xcode-select --install`
- App notarization needed for distribution outside App Store
- DMG creation: `hdiutil create -srcfolder dist/ PhotoAI.dmg`

**Linux:**
- GTK development libraries: `sudo apt install libgtk-3-dev`
- Different package formats: AppImage, deb, rpm
- Consider Flatpak/Snap for universal distribution

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run tests: `pytest tests/`
5. Submit a pull request

### Development Setup

**🚀 Quick Setup (Recommended):**
```bash
# Automated setup with virtual environment
python scripts/setup.py

# Platform-specific scripts
# Windows: scripts/setup.bat
# macOS/Linux: scripts/setup.sh

# Or using Makefile
make setup-full
```

**🔧 Manual Setup:**
```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -e ".[dev]"
# or: pip install -r requirements.txt

# 3. Verify installation
python -c "import photo_ai; print('✅ Setup successful!')"
```

**🛠️ Development Commands:**
```bash
# Run tests
pytest

# Format code
black photo_ai/

# Type checking
mypy photo_ai/

# Test GUI (requires display)
python photo_ai_gui.py

# Build application
make build-pyinstaller
```

### GUI Development
- Built with **PyQt6** for cross-platform compatibility
- Modern dark theme with professional styling
- Threaded processing to maintain UI responsiveness
- Modular widget system for easy extension

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Transformers**: Hugging Face transformers library for ML models
- **OpenCV**: Computer vision algorithms  
- **face_recognition**: Face detection capabilities
- **scikit-learn**: Machine learning utilities
- **PyQt6**: Cross-platform GUI framework

---

**Photo AI** - Making photo management intelligent and effortless with both powerful CLI tools and an intuitive desktop interface! 🚀
