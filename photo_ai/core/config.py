"""Configuration management for Photo AI."""

import os
from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Model configuration."""

    device: str = "auto"  # auto, cpu, cuda
    sharpness_model: str = "microsoft/resnet-50"
    feature_model: str = "google/vit-base-patch16-224-in21k"
    face_detection_model: str = "face_recognition"


@dataclass
class ProcessingConfig:
    """Processing configuration."""

    time_threshold: int = 300  # seconds
    cluster_eps: float = 0.4
    min_photos_to_cluster: int = 2
    sharpness_threshold: float = 0.7
    num_best_photos: int = 2


@dataclass
class VisaConfig:
    """Visa photo configuration."""

    dpi: int = 300
    photo_width_mm: int = 33
    photo_height_mm: int = 48
    face_height_ratio: float = 0.45
    face_top_margin_ratio: float = 0.15


@dataclass
class Config:
    """Main configuration class."""

    models: ModelConfig = field(default_factory=ModelConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    visa: VisaConfig = field(default_factory=VisaConfig)

    # Directories
    input_dir: str = "photos"
    good_dir: str = "photo-good"
    bad_dir: str = "photo-bad"
    output_dir: str = "output"

    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables."""
        config = cls()

        # Override with environment variables if present
        if os.getenv("PHOTO_AI_INPUT_DIR"):
            config.input_dir = os.getenv("PHOTO_AI_INPUT_DIR")
        if os.getenv("PHOTO_AI_GOOD_DIR"):
            config.good_dir = os.getenv("PHOTO_AI_GOOD_DIR")
        if os.getenv("PHOTO_AI_BAD_DIR"):
            config.bad_dir = os.getenv("PHOTO_AI_BAD_DIR")
        if os.getenv("PHOTO_AI_OUTPUT_DIR"):
            config.output_dir = os.getenv("PHOTO_AI_OUTPUT_DIR")

        return config

    def create_directories(self):
        """Create necessary directories."""
        for dir_path in [self.good_dir, self.bad_dir, self.output_dir]:
            os.makedirs(dir_path, exist_ok=True)
