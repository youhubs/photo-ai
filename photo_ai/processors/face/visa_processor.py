"""Visa photo processing - cropping and formatting for official documents."""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, Tuple
import os

from ...core.config import Config
from .detector import FaceDetector


class VisaPhotoProcessor:
    """Process photos for visa and official document requirements."""
    
    def __init__(self, config: Config):
        self.config = config
        self.face_detector = FaceDetector(config)
        
        # Calculate pixel dimensions from mm
        self.photo_width_px = int(self._mm_to_inches(config.visa.photo_width_mm) * config.visa.dpi)
        self.photo_height_px = int(self._mm_to_inches(config.visa.photo_height_mm) * config.visa.dpi)
    
    def _mm_to_inches(self, mm: float) -> float:
        """Convert millimeters to inches."""
        return mm / 25.4
    
    def _pixels_to_mm(self, pixels: int) -> float:
        """Convert pixels to millimeters."""
        return pixels / self.config.visa.dpi * 25.4
    
    def validate_input_image(self, image_path: str) -> Dict:
        """Validate if input image meets minimum requirements."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {'valid': False, 'reason': 'Could not read image file'}
            
            height, width = image.shape[:2]
            
            # Check minimum resolution
            min_width = 1600
            min_height = 1200
            if width < min_width or height < min_height:
                return {
                    'valid': False,
                    'reason': f'Image too small ({width}x{height}), need at least {min_width}x{min_height}'
                }
            
            # Check face requirements
            face_result = self.face_detector.analyze_face_quality(image_path)
            if face_result['quality_score'] < 0.3:
                return {
                    'valid': False,
                    'reason': f'Face quality issues: {", ".join(face_result["issues"])}'
                }
            
            return {'valid': True, 'face_result': face_result}
        except Exception as e:
            return {'valid': False, 'reason': str(e)}
    
    def crop_to_visa_dimensions(self, image_path: str) -> Tuple[np.ndarray, Dict]:
        """Crop image to exact visa photo dimensions."""
        face_result = self.face_detector.detect_with_margin(image_path)
        
        if not face_result.get('has_faces'):
            raise ValueError("No face detected for cropping")
        
        image = cv2.imread(image_path)
        largest_face = face_result['largest_face']
        
        # Calculate required scaling to fit face properly
        face_height = largest_face['height']
        target_face_height = int(self.photo_height_px * self.config.visa.face_height_ratio)
        scale = target_face_height / face_height
        
        # Resize image
        new_width = int(image.shape[1] * scale)
        new_height = int(image.shape[0] * scale)
        resized = cv2.resize(image, (new_width, new_height))
        
        # Calculate crop position
        face_top = int(largest_face['box'][0] * scale)
        face_center_x = int(largest_face['center_x'] * scale)
        
        # Position face according to visa requirements
        crop_top = face_top - int(self.photo_height_px * self.config.visa.face_top_margin_ratio)
        crop_left = face_center_x - self.photo_width_px // 2
        
        # Ensure crop bounds are valid
        crop_top = max(0, crop_top)
        crop_left = max(0, crop_left)
        
        # Add padding if needed
        padding_bottom = max(0, (crop_top + self.photo_height_px) - new_height)
        padding_right = max(0, (crop_left + self.photo_width_px) - new_width)
        
        if padding_bottom > 0 or padding_right > 0:
            resized = cv2.copyMakeBorder(
                resized, 0, padding_bottom, 0, padding_right,
                cv2.BORDER_CONSTANT, value=[255, 255, 255]
            )
        
        # Perform final crop
        cropped = resized[
            crop_top:crop_top + self.photo_height_px,
            crop_left:crop_left + self.photo_width_px
        ]
        
        # Ensure exact dimensions
        if cropped.shape[0] != self.photo_height_px or cropped.shape[1] != self.photo_width_px:
            cropped = cv2.resize(cropped, (self.photo_width_px, self.photo_height_px))
        
        crop_info = {
            'original_size': image.shape[:2],
            'scale_factor': scale,
            'crop_position': (crop_top, crop_left),
            'final_size': (self.photo_height_px, self.photo_width_px)
        }
        
        return cropped, crop_info
    
    def remove_background(self, image: np.ndarray) -> np.ndarray:
        """Remove background and replace with white."""
        try:
            h, w = image.shape[:2]
            
            # Create mask for GrabCut
            mask = np.zeros((h, w), np.uint8)
            
            # Assume face is in center region
            face_region_size = min(h, w) * 0.6
            center_x, center_y = w // 2, h // 2
            
            # Define probable foreground (face area)
            margin = int(face_region_size * 0.2)
            top = max(0, center_y - int(face_region_size // 2))
            bottom = min(h, center_y + int(face_region_size // 2))
            left = max(0, center_x - int(face_region_size // 2))
            right = min(w, center_x + int(face_region_size // 2))
            
            mask[top:bottom, left:right] = cv2.GC_PR_FGD
            
            # Define definite foreground (center of face)
            inner_top = center_y - margin
            inner_bottom = center_y + margin
            inner_left = center_x - margin
            inner_right = center_x + margin
            mask[inner_top:inner_bottom, inner_left:inner_right] = cv2.GC_FGD
            
            # Apply GrabCut
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            cv2.grabCut(image, mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
            
            # Create final mask
            final_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
            
            # Apply mask with white background
            result = image.copy()
            result[final_mask == 0] = [255, 255, 255]  # White background
            
            return result
        except Exception as e:
            print(f"Background removal failed: {e}, returning original image")
            return image
    
    def validate_output(self, image: np.ndarray) -> Dict:
        """Validate the final visa photo meets requirements."""
        issues = []
        
        # Check dimensions
        if image.shape[0] != self.photo_height_px or image.shape[1] != self.photo_width_px:
            issues.append(f"Incorrect dimensions: {image.shape[1]}x{image.shape[0]}")
        
        # Check if background is white
        corners = [
            image[0, 0], image[0, -1],
            image[-1, 0], image[-1, -1]
        ]
        
        for i, corner in enumerate(corners):
            if not np.allclose(corner, [255, 255, 255], atol=20):
                issues.append(f"Corner {i} not white: {corner}")
        
        # Check if there's a face in center region
        center_region = image[
            self.photo_height_px//4:3*self.photo_height_px//4,
            self.photo_width_px//4:3*self.photo_width_px//4
        ]
        
        if np.mean(center_region) > 240:  # Too white, likely no face
            issues.append("Center region too white - no face detected")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'dimensions_mm': (self.config.visa.photo_width_mm, self.config.visa.photo_height_mm),
            'dimensions_px': (self.photo_width_px, self.photo_height_px)
        }
    
    def process_visa_photo(self, input_path: str, output_path: str = None, debug: bool = False) -> Dict:
        """Complete visa photo processing pipeline."""
        try:
            # Validate input
            validation = self.validate_input_image(input_path)
            if not validation['valid']:
                return {'success': False, 'error': validation['reason']}
            
            # Crop to dimensions
            cropped_image, crop_info = self.crop_to_visa_dimensions(input_path)
            
            # Remove background
            final_image = self.remove_background(cropped_image)
            
            # Validate output
            output_validation = self.validate_output(final_image)
            
            # Save result
            if output_path:
                pil_image = Image.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
                pil_image.save(output_path, dpi=(self.config.visa.dpi, self.config.visa.dpi), quality=95)
            
            result = {
                'success': True,
                'output_path': output_path,
                'crop_info': crop_info,
                'validation': output_validation,
                'dimensions': {
                    'width_mm': self.config.visa.photo_width_mm,
                    'height_mm': self.config.visa.photo_height_mm,
                    'width_px': self.photo_width_px,
                    'height_px': self.photo_height_px,
                    'dpi': self.config.visa.dpi
                }
            }
            
            if debug:
                result['debug_info'] = self._create_debug_info(input_path, final_image, crop_info)
            
            return result
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _create_debug_info(self, input_path: str, processed_image: np.ndarray, crop_info: Dict) -> Dict:
        """Create debug information for troubleshooting."""
        return {
            'input_path': input_path,
            'processing_steps': [
                f"Original size: {crop_info['original_size']}",
                f"Scale factor: {crop_info['scale_factor']:.2f}",
                f"Crop position: {crop_info['crop_position']}",
                f"Final size: {crop_info['final_size']}"
            ],
            'output_stats': {
                'mean_brightness': np.mean(processed_image),
                'background_corners_white': all(
                    np.allclose(corner, [255, 255, 255], atol=20) 
                    for corner in [processed_image[0, 0], processed_image[0, -1], 
                                 processed_image[-1, 0], processed_image[-1, -1]]
                )
            }
        }