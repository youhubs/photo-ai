"""Photo AI - Advanced photo processing and analysis toolkit."""

__version__ = "1.0.0"
__author__ = "Photo AI Team"

from .core.photo_processor import PhotoProcessor
from .processors.quality.sharpness import SharpnessAnalyzer
from .processors.face.detector import FaceDetector
from .processors.background.remover import BackgroundRemover

# GUI components (optional import)
try:
    from .gui.app import PhotoAIApp
    from .gui.main_window import PhotoAIMainWindow
    GUI_AVAILABLE = True
    __all__ = [
        "PhotoProcessor",
        "SharpnessAnalyzer", 
        "FaceDetector",
        "BackgroundRemover",
        "PhotoAIApp",
        "PhotoAIMainWindow"
    ]
except ImportError:
    GUI_AVAILABLE = False
    __all__ = [
        "PhotoProcessor",
        "SharpnessAnalyzer", 
        "FaceDetector",
        "BackgroundRemover"
    ]