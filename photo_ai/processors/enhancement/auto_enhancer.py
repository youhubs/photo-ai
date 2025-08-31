"""Automatic photo enhancement for sports photography."""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AutoEnhancer:
    """Automatic photo enhancement for sports photos."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.enhancement_strength = self.config.get('enhancement_strength', 0.7)
    
    def enhance_photo(self, image_path: str, output_path: str) -> Dict:
        """
        Automatically enhance a sports photo.
        
        Args:
            image_path: Path to input image
            output_path: Path to save enhanced image
            
        Returns:
            Dict with enhancement results and statistics
        """
        try:
            # Load image
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                original_img = img.copy()
                
                # Analyze image characteristics
                analysis = self._analyze_image(img)
                
                # Apply enhancements based on analysis
                enhanced_img = self._apply_enhancements(img, analysis)
                
                # Save enhanced image
                enhanced_img.save(output_path, 'JPEG', quality=95, optimize=True)
                
                return {
                    'success': True,
                    'output_path': output_path,
                    'enhancements': analysis['enhancements_applied'],
                    'metrics': analysis['metrics'],
                    'improvements': self._calculate_improvements(original_img, enhanced_img)
                }
                
        except Exception as e:
            logger.error(f"Enhancement failed for {image_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'output_path': output_path
            }
    
    def _analyze_image(self, img: Image.Image) -> Dict:
        """Analyze image to determine needed enhancements."""
        # Convert to numpy for analysis
        img_array = np.array(img)
        
        # Calculate metrics
        brightness = np.mean(img_array)
        contrast = np.std(img_array)
        
        # Analyze histogram
        hist_r, hist_g, hist_b = [np.histogram(img_array[:,:,i], bins=256)[0] for i in range(3)]
        
        # Detect underexposure/overexposure
        dark_pixels = np.sum(img_array < 50) / img_array.size
        bright_pixels = np.sum(img_array > 205) / img_array.size
        
        analysis = {
            'metrics': {
                'brightness': brightness,
                'contrast': contrast,
                'dark_pixels_ratio': dark_pixels,
                'bright_pixels_ratio': bright_pixels
            },
            'enhancements_needed': [],
            'enhancements_applied': []
        }
        
        # Determine needed enhancements
        if brightness < 100:
            analysis['enhancements_needed'].append('brighten')
        elif brightness > 180:
            analysis['enhancements_needed'].append('darken')
            
        if contrast < 40:
            analysis['enhancements_needed'].append('increase_contrast')
            
        if dark_pixels > 0.3:
            analysis['enhancements_needed'].append('shadow_recovery')
            
        if bright_pixels > 0.1:
            analysis['enhancements_needed'].append('highlight_recovery')
        
        # Always apply slight sharpening for sports photos
        analysis['enhancements_needed'].append('sharpen')
        analysis['enhancements_needed'].append('color_enhancement')
        
        return analysis
    
    def _apply_enhancements(self, img: Image.Image, analysis: Dict) -> Image.Image:
        """Apply enhancements based on analysis."""
        enhanced = img.copy()
        applied = []
        
        for enhancement in analysis['enhancements_needed']:
            if enhancement == 'brighten':
                enhancer = ImageEnhance.Brightness(enhanced)
                enhanced = enhancer.enhance(1.2 * self.enhancement_strength + 0.3)
                applied.append('brightness_increase')
                
            elif enhancement == 'darken':
                enhancer = ImageEnhance.Brightness(enhanced)
                enhanced = enhancer.enhance(0.9 - 0.1 * self.enhancement_strength)
                applied.append('brightness_decrease')
                
            elif enhancement == 'increase_contrast':
                enhancer = ImageEnhance.Contrast(enhanced)
                enhanced = enhancer.enhance(1.3 * self.enhancement_strength + 0.7)
                applied.append('contrast_increase')
                
            elif enhancement == 'shadow_recovery':
                enhanced = self._recover_shadows(enhanced)
                applied.append('shadow_recovery')
                
            elif enhancement == 'highlight_recovery':
                enhanced = self._recover_highlights(enhanced)
                applied.append('highlight_recovery')
                
            elif enhancement == 'sharpen':
                enhanced = self._apply_sharpening(enhanced)
                applied.append('sharpening')
                
            elif enhancement == 'color_enhancement':
                enhanced = self._enhance_colors(enhanced)
                applied.append('color_enhancement')
        
        analysis['enhancements_applied'] = applied
        return enhanced
    
    def _recover_shadows(self, img: Image.Image) -> Image.Image:
        """Recover detail in shadow areas."""
        # Convert to numpy for processing
        img_array = np.array(img, dtype=np.float32)
        
        # Create shadow mask (dark areas)
        shadow_mask = (img_array < 80).any(axis=2)
        
        # Brighten shadow areas selectively
        for i in range(3):  # RGB channels
            channel = img_array[:,:,i]
            channel[shadow_mask] = np.clip(channel[shadow_mask] * 1.4, 0, 255)
            img_array[:,:,i] = channel
        
        return Image.fromarray(img_array.astype(np.uint8))
    
    def _recover_highlights(self, img: Image.Image) -> Image.Image:
        """Recover detail in highlight areas."""
        img_array = np.array(img, dtype=np.float32)
        
        # Create highlight mask (bright areas)
        highlight_mask = (img_array > 200).any(axis=2)
        
        # Darken highlight areas selectively
        for i in range(3):  # RGB channels
            channel = img_array[:,:,i]
            channel[highlight_mask] = np.clip(channel[highlight_mask] * 0.85, 0, 255)
            img_array[:,:,i] = channel
        
        return Image.fromarray(img_array.astype(np.uint8))
    
    def _apply_sharpening(self, img: Image.Image) -> Image.Image:
        """Apply intelligent sharpening for sports photos."""
        # Use unsharp mask filter for natural sharpening
        enhanced = img.filter(ImageFilter.UnsharpMask(
            radius=1.5,
            percent=150 * self.enhancement_strength + 50,
            threshold=3
        ))
        return enhanced
    
    def _enhance_colors(self, img: Image.Image) -> Image.Image:
        """Enhance color vibrancy for sports photos."""
        # Increase saturation slightly
        enhancer = ImageEnhance.Color(img)
        enhanced = enhancer.enhance(1.15 * self.enhancement_strength + 0.85)
        return enhanced
    
    def _calculate_improvements(self, original: Image.Image, enhanced: Image.Image) -> Dict:
        """Calculate improvement metrics."""
        orig_array = np.array(original)
        enh_array = np.array(enhanced)
        
        return {
            'brightness_change': np.mean(enh_array) - np.mean(orig_array),
            'contrast_change': np.std(enh_array) - np.std(orig_array),
            'color_enhancement': True  # Simplified for now
        }
    
    def detect_motion_blur(self, image_path: str) -> Dict:
        """
        Detect motion blur in sports photos.
        
        Returns:
            Dict with blur detection results
        """
        try:
            # Load image with OpenCV
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Calculate Laplacian variance (standard sharpness metric)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Calculate motion blur specific metrics
            motion_blur_score = self._calculate_motion_blur_score(gray)
            
            # Determine if image has motion blur
            is_motion_blurred = motion_blur_score > 0.6 and laplacian_var < 150
            
            return {
                'has_motion_blur': is_motion_blurred,
                'motion_blur_score': motion_blur_score,
                'sharpness_score': laplacian_var,
                'recommendation': 'discard' if is_motion_blurred else 'keep'
            }
            
        except Exception as e:
            logger.error(f"Motion blur detection failed for {image_path}: {e}")
            return {
                'has_motion_blur': False,
                'error': str(e),
                'recommendation': 'keep'  # Default to keep if detection fails
            }
    
    def _calculate_motion_blur_score(self, gray_img: np.ndarray) -> float:
        """Calculate motion blur score using directional gradients."""
        # Calculate gradients in different directions
        grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        # Motion blur typically shows strong directional patterns
        # Calculate variance in gradient directions
        direction_variance = np.var(direction[magnitude > np.percentile(magnitude, 75)])
        
        # Normalize to 0-1 range (higher = more motion blur)
        motion_score = 1.0 - min(direction_variance / (np.pi/2), 1.0)
        
        return motion_score
    
    def detect_poor_exposure(self, image_path: str) -> Dict:
        """
        Detect poor exposure (too dark or too bright) in photos.
        
        Returns:
            Dict with exposure analysis results
        """
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img_array = np.array(img)
                
                # Calculate exposure metrics
                mean_brightness = np.mean(img_array)
                brightness_std = np.std(img_array)
                
                # Calculate histogram
                hist, _ = np.histogram(img_array.flatten(), bins=256, range=[0, 256])
                
                # Calculate clipping (loss of detail)
                dark_clipping = np.sum(hist[:10]) / img_array.size  # Very dark pixels
                bright_clipping = np.sum(hist[245:]) / img_array.size  # Very bright pixels
                
                # Determine exposure quality
                is_underexposed = mean_brightness < 60 or dark_clipping > 0.4
                is_overexposed = mean_brightness > 200 or bright_clipping > 0.3
                has_poor_exposure = is_underexposed or is_overexposed
                
                return {
                    'has_poor_exposure': has_poor_exposure,
                    'is_underexposed': is_underexposed,
                    'is_overexposed': is_overexposed,
                    'brightness_score': mean_brightness,
                    'dark_clipping': dark_clipping,
                    'bright_clipping': bright_clipping,
                    'recommendation': 'discard' if has_poor_exposure else 'keep'
                }
                
        except Exception as e:
            logger.error(f"Exposure detection failed for {image_path}: {e}")
            return {
                'has_poor_exposure': False,
                'error': str(e),
                'recommendation': 'keep'
            }