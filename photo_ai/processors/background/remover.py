"""Background removal utilities."""

import cv2
import numpy as np
from typing import Tuple, Optional
from PIL import Image

from ...core.config import Config


class BackgroundRemover:
    """Remove and replace image backgrounds."""

    def __init__(self, config: Config):
        self.config = config

    def remove_with_grabcut(
        self, image: np.ndarray, foreground_box: Optional[Tuple] = None
    ) -> np.ndarray:
        """Remove background using GrabCut algorithm."""
        try:
            h, w = image.shape[:2]
            mask = np.zeros((h, w), np.uint8)

            if foreground_box:
                # Use provided bounding box
                x, y, width, height = foreground_box
                rect = (x, y, width, height)
                cv2.grabCut(image, mask, rect, None, None, 5, cv2.GC_INIT_WITH_RECT)
            else:
                # Estimate center region as foreground
                center_x, center_y = w // 2, h // 2
                size = min(w, h) // 3

                # Set probable foreground
                mask[center_y - size : center_y + size, center_x - size : center_x + size] = (
                    cv2.GC_PR_FGD
                )

                # Set definite foreground (smaller center region)
                inner_size = size // 2
                mask[
                    center_y - inner_size : center_y + inner_size,
                    center_x - inner_size : center_x + inner_size,
                ] = cv2.GC_FGD

                bgd_model = np.zeros((1, 65), np.float64)
                fgd_model = np.zeros((1, 65), np.float64)
                cv2.grabCut(image, mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)

            # Create final mask
            final_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(
                "uint8"
            )

            return final_mask
        except Exception as e:
            print(f"GrabCut failed: {e}")
            # Return full foreground mask as fallback
            return np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)

    def remove_with_edge_detection(self, image: np.ndarray) -> np.ndarray:
        """Remove background using edge detection and contour analysis."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Create mask from largest contour (assumed to be main subject)
            mask = np.zeros(gray.shape, dtype=np.uint8)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                cv2.fillPoly(mask, [largest_contour], 255)

            # Dilate to include more of the subject
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)

            return mask // 255  # Convert to 0-1 values
        except Exception as e:
            print(f"Edge detection method failed: {e}")
            return np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)

    def apply_mask_with_background(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        background_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        """Apply mask to image with specified background color."""
        try:
            # Ensure mask is the right shape
            if len(mask.shape) == 2:
                mask = np.stack([mask] * 3, axis=2)

            # Create background
            background = np.full_like(image, background_color, dtype=np.uint8)

            # Apply mask
            result = image * mask + background * (1 - mask)

            return result.astype(np.uint8)
        except Exception as e:
            print(f"Mask application failed: {e}")
            return image

    def remove_background_comprehensive(
        self,
        image: np.ndarray,
        method: str = "grabcut",
        background_color: Tuple[int, int, int] = (255, 255, 255),
        foreground_hint: Optional[Tuple] = None,
    ) -> np.ndarray:
        """Comprehensive background removal with multiple methods."""

        if method == "grabcut":
            mask = self.remove_with_grabcut(image, foreground_hint)
        elif method == "edge":
            mask = self.remove_with_edge_detection(image)
        else:
            raise ValueError(f"Unknown method: {method}")

        return self.apply_mask_with_background(image, mask, background_color)

    def auto_detect_background_color(self, image: np.ndarray) -> Tuple[int, int, int]:
        """Automatically detect the dominant background color."""
        try:
            h, w = image.shape[:2]

            # Sample edges of image (assumed to be background)
            edge_pixels = []

            # Top and bottom edges
            edge_pixels.extend(image[0, :].reshape(-1, 3))
            edge_pixels.extend(image[-1, :].reshape(-1, 3))

            # Left and right edges
            edge_pixels.extend(image[:, 0].reshape(-1, 3))
            edge_pixels.extend(image[:, -1].reshape(-1, 3))

            edge_pixels = np.array(edge_pixels)

            # Find most common color
            unique_colors, counts = np.unique(
                edge_pixels.reshape(-1, 3), axis=0, return_counts=True
            )
            dominant_color = unique_colors[np.argmax(counts)]

            return tuple(dominant_color)
        except Exception as e:
            print(f"Auto background detection failed: {e}")
            return (255, 255, 255)  # Default to white

    def remove_background_smart(self, image: np.ndarray) -> np.ndarray:
        """Smart background removal with automatic method selection."""
        # Try GrabCut first
        try:
            result = self.remove_background_comprehensive(image, method="grabcut")

            # Validate result - check if background was actually removed
            corners = [result[0, 0], result[0, -1], result[-1, 0], result[-1, -1]]
            if all(np.allclose(corner, [255, 255, 255], atol=30) for corner in corners):
                return result
        except Exception as e:
            print(f"GrabCut method failed: {e}")

        # Fallback to edge detection
        try:
            return self.remove_background_comprehensive(image, method="edge")
        except Exception as e:
            print(f"All background removal methods failed: {e}")
            return image  # Return original if all methods fail
