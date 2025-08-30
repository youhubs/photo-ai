"""Sharpness analysis and blur detection."""

import cv2
import torch
import numpy as np
from typing import Dict, List, Tuple
from PIL import Image
from transformers import AutoFeatureExtractor, ResNetForImageClassification

from ...core.config import Config


class SharpnessAnalyzer:
    """Analyze image sharpness using multiple methods."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.models.device != "cpu" else "cpu")
        self._init_models()
    
    def _init_models(self):
        """Initialize ML models for sharpness analysis."""
        try:
            self.extractor = AutoFeatureExtractor.from_pretrained(self.config.models.sharpness_model)
            self.model = ResNetForImageClassification.from_pretrained(
                self.config.models.sharpness_model
            ).to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Warning: Could not load ML model for sharpness: {e}")
            self.extractor = None
            self.model = None
    
    def analyze_laplacian_variance(self, image_path: str, threshold: float = 100.0) -> Dict:
        """Analyze sharpness using Laplacian variance method."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not load image")
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            return {
                'method': 'laplacian_variance',
                'score': variance,
                'is_sharp': variance > threshold,
                'threshold': threshold
            }
        except Exception as e:
            return {'method': 'laplacian_variance', 'error': str(e)}
    
    def analyze_ml_based(self, image_path: str) -> Dict:
        """Analyze sharpness using pre-trained ML model."""
        if not self.model or not self.extractor:
            return {'method': 'ml_based', 'error': 'Model not available'}
        
        try:
            image = Image.open(image_path)
            inputs = self.extractor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                sharp_prob = probs[0][1].item()  # Assuming class 1 is sharp
            
            return {
                'method': 'ml_based',
                'score': sharp_prob,
                'is_sharp': sharp_prob > self.config.processing.sharpness_threshold,
                'threshold': self.config.processing.sharpness_threshold
            }
        except Exception as e:
            return {'method': 'ml_based', 'error': str(e)}
    
    def analyze_gradient_based(self, image_path: str, threshold: float = 50.0) -> Dict:
        """Analyze sharpness using gradient-based method."""
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError("Could not load image")
            
            # Calculate gradients
            grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate gradient magnitude
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            score = np.mean(magnitude)
            
            return {
                'method': 'gradient_based',
                'score': score,
                'is_sharp': score > threshold,
                'threshold': threshold
            }
        except Exception as e:
            return {'method': 'gradient_based', 'error': str(e)}
    
    def analyze_comprehensive(self, image_path: str) -> Dict:
        """Comprehensive sharpness analysis using multiple methods."""
        results = {
            'image_path': image_path,
            'analyses': []
        }
        
        # Run all analysis methods
        methods = [
            self.analyze_laplacian_variance,
            self.analyze_ml_based,
            self.analyze_gradient_based
        ]
        
        for method in methods:
            result = method(image_path)
            results['analyses'].append(result)
        
        # Calculate overall assessment
        valid_results = [r for r in results['analyses'] if 'error' not in r]
        if valid_results:
            sharp_count = sum(1 for r in valid_results if r.get('is_sharp', False))
            results['overall_is_sharp'] = sharp_count >= len(valid_results) / 2
            results['confidence'] = sharp_count / len(valid_results)
        else:
            results['overall_is_sharp'] = False
            results['confidence'] = 0.0
        
        return results
    
    def batch_analyze(self, image_paths: List[str]) -> Dict[str, Dict]:
        """Analyze multiple images for sharpness."""
        results = {}
        for path in image_paths:
            results[path] = self.analyze_comprehensive(path)
        return results