"""Photo AI - Advanced photo processing and analysis toolkit."""

__version__ = "1.0.0"
__author__ = "Photo AI Team"

from .core.photo_processor import PhotoProcessor
from .processors.quality.sharpness import SharpnessAnalyzer
from .processors.face.detector import FaceDetector
from .processors.background.remover import BackgroundRemover

__all__ = [
    "PhotoProcessor",
    "SharpnessAnalyzer", 
    "FaceDetector",
    "BackgroundRemover"
]