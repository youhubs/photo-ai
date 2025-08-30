"""Face detection and analysis."""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import face_recognition
from PIL import Image

from ...core.config import Config


class FaceDetector:
    """Detect and analyze faces in images."""

    def __init__(self, config: Config):
        self.config = config

    def detect_faces(self, image_path: str) -> Dict:
        """Detect all faces in an image."""
        try:
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)

            if not face_locations:
                return {"image_path": image_path, "face_count": 0, "faces": [], "has_faces": False}

            faces = []
            for i, (top, right, bottom, left) in enumerate(face_locations):
                face_info = {
                    "id": i,
                    "box": (top, right, bottom, left),
                    "width": right - left,
                    "height": bottom - top,
                    "area": (right - left) * (bottom - top),
                    "center_x": (left + right) // 2,
                    "center_y": (top + bottom) // 2,
                }
                faces.append(face_info)

            # Find largest face
            largest_face = max(faces, key=lambda f: f["area"]) if faces else None

            return {
                "image_path": image_path,
                "face_count": len(faces),
                "faces": faces,
                "largest_face": largest_face,
                "has_faces": len(faces) > 0,
                "image_shape": image.shape,
            }
        except Exception as e:
            return {"image_path": image_path, "error": str(e), "has_faces": False, "face_count": 0}

    def detect_with_margin(self, image_path: str, margin_ratio: float = 0.25) -> Dict:
        """Detect faces with safety margins."""
        result = self.detect_faces(image_path)

        if not result.get("has_faces"):
            return result

        # Add margins to all detected faces
        image_height, image_width = result["image_shape"][:2]

        for face in result["faces"]:
            top, right, bottom, left = face["box"]
            face_width = right - left
            face_height = bottom - top

            # Calculate margin
            margin = int(max(face_width, face_height) * margin_ratio)

            # Apply margin with bounds checking
            face["box_with_margin"] = (
                max(0, top - margin),
                min(image_width, right + margin),
                min(image_height, bottom + margin),
                max(0, left - margin),
            )

        return result

    def validate_face_for_visa(self, face_info: Dict, image_shape: Tuple) -> Dict:
        """Validate if face meets visa photo requirements."""
        if not face_info:
            return {"valid": False, "reason": "No face detected"}

        image_height = image_shape[0]
        face_height = face_info["height"]
        face_width = face_info["width"]

        # Check face size relative to image
        face_height_ratio = face_height / image_height
        if face_height_ratio < 0.25:
            return {
                "valid": False,
                "reason": f"Face too small ({face_height_ratio*100:.1f}% of image height, need >25%)",
            }

        if face_height_ratio > 0.45:
            return {
                "valid": False,
                "reason": f"Face too large ({face_height_ratio*100:.1f}% of image height, need <45%)",
            }

        # Check aspect ratio (face should not be too wide or narrow)
        aspect_ratio = face_width / face_height
        if aspect_ratio < 0.6 or aspect_ratio > 1.4:
            return {"valid": False, "reason": f"Face aspect ratio unusual ({aspect_ratio:.2f})"}

        # Check position (face should be roughly centered horizontally)
        image_width = image_shape[1]
        face_center_x = face_info["center_x"]
        horizontal_offset = abs(face_center_x - image_width // 2) / image_width

        if horizontal_offset > 0.3:
            return {
                "valid": False,
                "reason": f"Face not centered horizontally (offset: {horizontal_offset*100:.1f}%)",
            }

        return {
            "valid": True,
            "face_height_ratio": face_height_ratio,
            "aspect_ratio": aspect_ratio,
            "horizontal_offset": horizontal_offset,
        }

    def get_face_landmarks(self, image_path: str) -> Dict:
        """Get detailed face landmarks."""
        try:
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)
            face_landmarks = face_recognition.face_landmarks(image, face_locations)

            return {
                "image_path": image_path,
                "face_count": len(face_locations),
                "landmarks": face_landmarks,
                "has_landmarks": len(face_landmarks) > 0,
            }
        except Exception as e:
            return {"image_path": image_path, "error": str(e), "has_landmarks": False}

    def analyze_face_quality(self, image_path: str) -> Dict:
        """Comprehensive face quality analysis."""
        face_result = self.detect_with_margin(image_path)

        if not face_result.get("has_faces"):
            return {"image_path": image_path, "quality_score": 0.0, "issues": ["No face detected"]}

        largest_face = face_result["largest_face"]
        image_shape = face_result["image_shape"]

        # Validate for visa requirements
        visa_validation = self.validate_face_for_visa(largest_face, image_shape)

        issues = []
        quality_score = 1.0

        if not visa_validation["valid"]:
            issues.append(visa_validation["reason"])
            quality_score *= 0.3

        # Check for multiple faces (not ideal for visa photos)
        if face_result["face_count"] > 1:
            issues.append(f"Multiple faces detected ({face_result['face_count']})")
            quality_score *= 0.7

        # Check face area (larger faces generally better)
        total_pixels = image_shape[0] * image_shape[1]
        face_area_ratio = largest_face["area"] / total_pixels

        if face_area_ratio < 0.05:
            issues.append("Face area very small")
            quality_score *= 0.5
        elif face_area_ratio > 0.5:
            issues.append("Face area very large")
            quality_score *= 0.8

        return {
            "image_path": image_path,
            "quality_score": quality_score,
            "issues": issues,
            "face_info": largest_face,
            "visa_validation": visa_validation,
        }
