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

    # Player recognition settings
    face_match_threshold: float = (
        0.5  # Threshold for matching faces to reference players (0.5 = more permissive for sports photos)
    )
    visual_similarity_threshold: float = (
        0.7  # Threshold for matching non-face photos using visual features
    )
    enable_non_face_matching: bool = True  # Enable matching of photos without detectable faces
    enable_jersey_number_matching: bool = (
        False  # Enable jersey number detection and matching (OCR-based, works best with clear back view photos)
    )
    jersey_number_confidence_threshold: float = 0.6  # Minimum confidence for OCR number detection

    # Performance optimization settings
    use_parallel_processing: bool = (
        False  # Enable parallel processing for faster performance (disabled by default due to thread safety issues with face_recognition)
    )
    max_worker_threads: int = 4  # Maximum number of worker threads for parallel processing
    fast_mode: bool = True  # Use faster algorithms with slight accuracy trade-off
    batch_size: int = 10  # Process photos in batches to optimize memory usage


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
