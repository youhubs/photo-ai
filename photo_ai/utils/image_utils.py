"""Image utility functions."""

import os
import glob
from typing import List
from PIL import Image
import numpy as np


def load_image(path: str) -> Image.Image:
    """Load an image from file path."""
    try:
        return Image.open(path)
    except Exception as e:
        raise ValueError(f"Could not load image {path}: {str(e)}")


def save_image(image: Image.Image, path: str, quality: int = 95, dpi: tuple = (300, 300)):
    """Save image with specified quality and DPI."""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        image.save(path, quality=quality, dpi=dpi)
    except Exception as e:
        raise ValueError(f"Could not save image to {path}: {str(e)}")


def get_image_paths(directory: str, extensions: List[str] = None) -> List[str]:
    """Get all image file paths from directory."""
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff']
    
    image_paths = []
    if os.path.isdir(directory):
        for file in os.listdir(directory):
            if any(file.lower().endswith(ext) for ext in extensions):
                image_paths.append(os.path.join(directory, file))
    
    return sorted(image_paths)


def resize_image(image: Image.Image, max_size: tuple = None, maintain_aspect: bool = True) -> Image.Image:
    """Resize image while optionally maintaining aspect ratio."""
    if max_size is None:
        return image
    
    if maintain_aspect:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        return image
    else:
        return image.resize(max_size, Image.Resampling.LANCZOS)


def convert_to_rgb(image: Image.Image) -> Image.Image:
    """Convert image to RGB format."""
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image