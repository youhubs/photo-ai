"""Sharpness analysis and blur detection - macOS safe."""

import numpy as np
from PIL import Image, ImageFilter
import os
import logging
from typing import Dict, List
import gc

logger = logging.getLogger(__name__)


# For backward compatibility, keep the original class name
class SharpnessAnalyzer:
    """Sharpness analysis using only PIL and numpy - completely safe for macOS."""

    def __init__(self, config):
        self.config = config
        self.laplacian_threshold = getattr(config.processing, "sharpness_threshold", 100.0)
        self.gradient_threshold = 50.0

    def _calculate_laplacian_pil(self, image_path: str) -> float:
        """Calculate Laplacian variance using only PIL."""
        try:
            with Image.open(image_path) as img:
                # Convert to grayscale
                if img.mode != "L":
                    img = img.convert("L")

                # Apply Laplacian filter manually
                laplacian_kernel = ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1)
                laplacian = img.filter(laplacian_kernel)

                # Convert to numpy array and calculate variance
                laplacian_array = np.array(laplacian, dtype=np.float64)
                variance = np.var(laplacian_array)

                return float(variance)
        except Exception as e:
            logger.error(f"Laplacian calculation failed: {e}")
            return 0.0

    def _calculate_gradient_pil(self, image_path: str) -> float:
        """Calculate gradient magnitude using only PIL."""
        try:
            with Image.open(image_path) as img:
                if img.mode != "L":
                    img = img.convert("L")

                # Sobel operators using PIL filters
                sobel_x_kernel = ImageFilter.Kernel((3, 3), [-1, 0, 1, -2, 0, 2, -1, 0, 1], scale=1)
                sobel_y_kernel = ImageFilter.Kernel((3, 3), [-1, -2, -1, 0, 0, 0, 1, 2, 1], scale=1)

                grad_x = img.filter(sobel_x_kernel)
                grad_y = img.filter(sobel_y_kernel)

                # Convert to numpy for magnitude calculation
                grad_x_array = np.array(grad_x, dtype=np.float64)
                grad_y_array = np.array(grad_y, dtype=np.float64)

                magnitude = np.sqrt(grad_x_array**2 + grad_y_array**2)
                mean_magnitude = np.mean(magnitude)

                return float(mean_magnitude)
        except Exception as e:
            logger.error(f"Gradient calculation failed: {e}")
            return 0.0

    def analyze_comprehensive(self, image_path: str) -> Dict:
        """Run all sharpness checks safely."""
        try:
            # Method 1: Laplacian variance
            laplacian_score = self._calculate_laplacian_pil(image_path)

            # Method 2: Gradient magnitude
            gradient_score = self._calculate_gradient_pil(image_path)

            # Method 3: Edge count (simple alternative)
            edge_count = self._count_edges_simple(image_path)

            # Combine scores
            is_sharp = (
                laplacian_score > self.laplacian_threshold
                or gradient_score > self.gradient_threshold
                or edge_count > 5000  # Arbitrary threshold
            )

            confidence = min(1.0, (laplacian_score / 500.0 + gradient_score / 100.0) / 2)

            return {
                "image_path": image_path,
                "analyses": [
                    {
                        "method": "laplacian_variance",
                        "score": laplacian_score,
                        "is_sharp": laplacian_score > self.laplacian_threshold,
                    },
                    {
                        "method": "gradient_based",
                        "score": gradient_score,
                        "is_sharp": gradient_score > self.gradient_threshold,
                    },
                    {"method": "edge_count", "score": edge_count, "is_sharp": edge_count > 5000},
                ],
                "overall_is_sharp": is_sharp,
                "confidence": confidence,
                "error": None,
            }

        except Exception as e:
            logger.error(f"Sharpness analysis failed for {image_path}: {e}")
            return {
                "image_path": image_path,
                "analyses": [],
                "overall_is_sharp": False,
                "confidence": 0.0,
                "error": str(e),
            }

    def _count_edges_simple(self, image_path: str) -> int:
        """Simple edge counting method."""
        try:
            with Image.open(image_path) as img:
                if img.mode != "L":
                    img = img.convert("L")

                # Simple edge detection using difference from median
                img_array = np.array(img, dtype=np.float64)
                median = np.median(img_array)
                edges = np.abs(img_array - median) > 30  # Threshold

                return int(np.sum(edges))
        except Exception as e:
            logger.error(f"Edge counting failed: {e}")
            return 0

    def batch_analyze(self, image_paths: List[str]) -> Dict[str, Dict]:
        """Analyze multiple images safely."""
        results = {}

        for i, path in enumerate(image_paths, 1):
            try:
                logger.info(f"Analyzing sharpness {i}/{len(image_paths)}: {os.path.basename(path)}")
                results[path] = self.analyze_comprehensive(path)

                # Force garbage collection every few images
                if i % 3 == 0:
                    gc.collect()

            except Exception as e:
                logger.error(f"Error analyzing {path}: {e}")
                results[path] = {
                    "image_path": path,
                    "analyses": [],
                    "overall_is_sharp": False,
                    "confidence": 0.0,
                    "error": str(e),
                }

        return results


# Alias for backward compatibility
SafeSharpnessAnalyzer = SharpnessAnalyzer
