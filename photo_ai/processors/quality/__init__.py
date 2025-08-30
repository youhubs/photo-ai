"""Quality assessment processors."""

from .sharpness import SharpnessAnalyzer
from .duplicates import DuplicateDetector

__all__ = ["SharpnessAnalyzer", "DuplicateDetector"]
