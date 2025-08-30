# Photo AI 📸🤖

Advanced photo processing and analysis toolkit powered by machine learning. Automatically organize, enhance, and process your photos with intelligent algorithms.

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

# Install the package
pip install -e .

# Or install dependencies manually
pip install torch torchvision transformers opencv-python face-recognition pillow numpy scikit-learn matplotlib
```

### Basic Usage

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

## Command Line Interface

### `photo-ai process [directory]`
Run the complete photo processing pipeline:
- Analyze sharpness and move photos to good/bad folders
- Detect duplicates and similar photos
- Select best photos from each group
- Generate processing statistics

```bash
# Process photos in current directory
photo-ai process

# Process specific directory
photo-ai process /path/to/photos

# Use custom output directories
photo-ai process photos/ --good-dir sharp --bad-dir blurry
```

### `photo-ai visa <input> [output]`
Create visa/passport photos that meet official requirements:
- Precise 33x48mm dimensions at 300 DPI
- Proper face positioning and sizing
- White background removal
- Compliance validation

```bash
# Create visa photo
photo-ai visa portrait.jpg visa.jpg

# Enable debug mode for troubleshooting
photo-ai visa portrait.jpg visa.jpg --debug
```

### `photo-ai analyze <directory>`
Analyze photo quality without moving files:
- Sharpness assessment
- Duplicate detection
- Face quality analysis
- Export results in JSON or text format

```bash
# Analyze and show text summary
photo-ai analyze photos/

# Export detailed JSON report
photo-ai analyze photos/ --format json > report.json
```

### `photo-ai stats`
Show processing statistics and directory information:
- File counts in each category
- Directory locations and status
- Processing history

## Configuration

### Environment Variables
```bash
export PHOTO_AI_INPUT_DIR="/path/to/input"
export PHOTO_AI_GOOD_DIR="/path/to/good"
export PHOTO_AI_BAD_DIR="/path/to/bad"  
export PHOTO_AI_OUTPUT_DIR="/path/to/output"
```

### Command Line Options
```bash
photo-ai --input-dir photos/ --good-dir sharp --bad-dir blurry --output-dir processed/
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
from photo_ai.core.config import Config, ProcessingConfig, VisaConfig

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
└── cli/                  # Command line interface
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

**GPU/Memory Issues**
- Reduce batch size for large photo collections
- Use CPU-only mode: set device to "cpu" in config
- Close other applications to free memory

### Debug Mode
Enable debug output for troubleshooting:
```bash
photo-ai visa input.jpg output.jpg --debug
photo-ai process photos/ --verbose
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run tests: `pytest tests/`
5. Submit a pull request

### Development Setup
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black photo_ai/

# Type checking
mypy photo_ai/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Transformers**: Hugging Face transformers library for ML models
- **OpenCV**: Computer vision algorithms
- **face_recognition**: Face detection capabilities
- **scikit-learn**: Machine learning utilities

---

**Photo AI** - Making photo management intelligent and effortless! 🚀