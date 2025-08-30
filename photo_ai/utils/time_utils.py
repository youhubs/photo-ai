"""Time utility functions."""

import os
from datetime import datetime
from typing import Optional
from PIL import Image


def get_capture_time(image_path: str) -> Optional[datetime]:
    """Extract capture time from image EXIF data."""
    try:
        with Image.open(image_path) as img:
            exif = img._getexif()
            if exif and 36867 in exif:  # DateTimeOriginal
                return datetime.strptime(exif[36867], "%Y:%m:%d %H:%M:%S")
            elif exif and 306 in exif:  # DateTime
                return datetime.strptime(exif[306], "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    
    # Fall back to file modification time
    try:
        timestamp = os.path.getmtime(image_path)
        return datetime.fromtimestamp(timestamp)
    except Exception:
        return None


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"