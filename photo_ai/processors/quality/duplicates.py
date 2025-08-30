"""Duplicate photo detection using perceptual hashing and clustering."""

import os
import cv2
import numpy as np
from typing import Dict, List, Set, Tuple
from PIL import Image
import torch
from sklearn.cluster import DBSCAN
from transformers import ViTImageProcessor, ViTModel

from ...core.config import Config
from ...utils.time_utils import get_capture_time


class DuplicateDetector:
    """Detect duplicate and similar photos."""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and config.models.device != "cpu" else "cpu"
        )
        self._init_models()

    def _init_models(self):
        """Initialize feature extraction models."""
        try:
            self.processor = ViTImageProcessor.from_pretrained(self.config.models.feature_model)
            self.model = ViTModel.from_pretrained(self.config.models.feature_model).to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Warning: Could not load feature extraction model: {e}")
            self.processor = None
            self.model = None

    def calculate_perceptual_hash(self, image_path: str, hash_size: int = 8) -> str:
        """Calculate perceptual hash for an image."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not load image")

            # Resize and convert to grayscale
            resized = cv2.resize(image, (hash_size + 1, hash_size))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            # Calculate horizontal gradient
            diff = gray[:, 1:] > gray[:, :-1]

            # Convert to hash string
            return "".join(str(b) for b in diff.flatten().astype(int))
        except Exception as e:
            return f"error_{hash(image_path)}"

    def extract_deep_features(self, image_path: str) -> np.ndarray:
        """Extract deep features using ViT model."""
        if not self.model or not self.processor:
            # Fallback to simple color histogram
            return self._extract_color_histogram(image_path)

        try:
            image = Image.open(image_path)
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

            return features.flatten()
        except Exception as e:
            print(f"Feature extraction failed for {image_path}: {e}")
            return self._extract_color_histogram(image_path)

    def _extract_color_histogram(self, image_path: str) -> np.ndarray:
        """Fallback feature extraction using color histogram."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not load image")

            # Calculate histograms for each channel
            hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
            hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])

            # Normalize and concatenate
            hist = np.concatenate([hist_b, hist_g, hist_r]).flatten()
            return hist / (hist.sum() + 1e-7)
        except Exception as e:
            # Return zero vector if all else fails
            return np.zeros(768)

    def find_duplicates_by_hash(self, image_paths: List[str]) -> Dict[str, List[str]]:
        """Find exact duplicates using perceptual hashing."""
        hash_groups = {}

        for path in image_paths:
            hash_val = self.calculate_perceptual_hash(path)
            if hash_val not in hash_groups:
                hash_groups[hash_val] = []
            hash_groups[hash_val].append(path)

        # Return only groups with multiple images
        return {h: paths for h, paths in hash_groups.items() if len(paths) > 1}

    def find_similar_by_clustering(self, image_paths: List[str]) -> Dict[int, List[str]]:
        """Find similar images using feature clustering."""
        if len(image_paths) < 2:
            return {}

        # Extract features for all images
        features = []
        valid_paths = []

        for path in image_paths:
            try:
                feature = self.extract_deep_features(path)
                features.append(feature)
                valid_paths.append(path)
            except Exception as e:
                print(f"Skipping {path} due to feature extraction error: {e}")

        if len(features) < 2:
            return {}

        # Normalize features
        features = np.array(features)
        features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-7)

        # Cluster similar images
        clustering = DBSCAN(
            eps=self.config.processing.cluster_eps, min_samples=2, metric="cosine"
        ).fit(features)

        # Group by cluster labels
        clusters = {}
        for idx, label in enumerate(clustering.labels_):
            if label != -1:  # Ignore noise points
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(valid_paths[idx])

        return clusters

    def find_time_based_groups(
        self, image_paths: List[str], time_threshold: int = None
    ) -> List[List[str]]:
        """Group images taken within a time threshold."""
        if time_threshold is None:
            time_threshold = self.config.processing.time_threshold

        # Get timestamps for all images
        image_times = []
        for path in image_paths:
            timestamp = get_capture_time(path)
            if timestamp:
                image_times.append((path, timestamp))

        # Sort by time
        image_times.sort(key=lambda x: x[1])

        # Group by time proximity
        groups = []
        current_group = []

        for path, timestamp in image_times:
            if not current_group:
                current_group = [path]
            else:
                # Check if within threshold of the first image in current group
                first_timestamp = next(t for p, t in image_times if p == current_group[0])
                if (timestamp - first_timestamp).total_seconds() <= time_threshold:
                    current_group.append(path)
                else:
                    if len(current_group) > 1:
                        groups.append(current_group)
                    current_group = [path]

        # Don't forget the last group
        if len(current_group) > 1:
            groups.append(current_group)

        return groups

    def find_comprehensive_duplicates(self, image_paths: List[str]) -> Dict:
        """Comprehensive duplicate detection using all methods."""
        results = {
            "total_images": len(image_paths),
            "exact_duplicates": self.find_duplicates_by_hash(image_paths),
            "similar_clusters": self.find_similar_by_clustering(image_paths),
            "time_based_groups": self.find_time_based_groups(image_paths),
        }

        # Calculate statistics
        exact_duplicate_count = sum(len(group) for group in results["exact_duplicates"].values())
        similar_images_count = sum(len(group) for group in results["similar_clusters"].values())

        results["stats"] = {
            "exact_duplicates_count": exact_duplicate_count,
            "similar_images_count": similar_images_count,
            "unique_images_estimate": len(image_paths) - exact_duplicate_count,
        }

        return results
