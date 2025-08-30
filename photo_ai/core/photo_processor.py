"""Main photo processing orchestrator."""

import os
import shutil
from typing import Dict, List, Optional
from pathlib import Path

from .config import Config
from ..processors.quality.sharpness import SharpnessAnalyzer
from ..processors.quality.duplicates import DuplicateDetector
from ..processors.face.detector import FaceDetector
from ..processors.face.visa_processor import VisaPhotoProcessor
from ..utils.image_utils import get_image_paths


class PhotoProcessor:
    """Main photo processing orchestrator."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config.from_env()
        self.config.create_directories()
        
        # Initialize processors
        self.sharpness_analyzer = SharpnessAnalyzer(self.config)
        self.duplicate_detector = DuplicateDetector(self.config)
        self.face_detector = FaceDetector(self.config)
        self.visa_processor = VisaPhotoProcessor(self.config)
    
    def process_photos_pipeline(self, input_dir: Optional[str] = None) -> Dict:
        """Run complete photo processing pipeline."""
        input_dir = input_dir or self.config.input_dir
        
        if not os.path.exists(input_dir):
            return {'success': False, 'error': f'Input directory does not exist: {input_dir}'}
        
        print(f"Processing photos from: {input_dir}")
        
        # Get all image paths
        image_paths = get_image_paths(input_dir)
        if not image_paths:
            return {'success': False, 'error': 'No images found in input directory'}
        
        print(f"Found {len(image_paths)} images")
        
        results = {
            'input_dir': input_dir,
            'total_images': len(image_paths),
            'stages': {}
        }
        
        # Stage 1: Sharpness Analysis
        print("\\nStage 1: Analyzing sharpness...")
        sharpness_results = self.sharpness_analyzer.batch_analyze(image_paths)
        sharp_images = [path for path, result in sharpness_results.items() 
                       if result.get('overall_is_sharp', False)]
        
        # Move images to appropriate directories
        self._organize_by_sharpness(sharpness_results)
        
        results['stages']['sharpness'] = {
            'processed': len(sharpness_results),
            'sharp': len(sharp_images),
            'blurry': len(image_paths) - len(sharp_images)
        }
        
        print(f"Sharp images: {len(sharp_images)}, Blurry: {len(image_paths) - len(sharp_images)}")
        
        # Stage 2: Duplicate Detection
        if len(sharp_images) >= self.config.processing.min_photos_to_cluster:
            print("\\nStage 2: Detecting duplicates and clustering...")
            duplicate_results = self.duplicate_detector.find_comprehensive_duplicates(sharp_images)
            results['stages']['duplicates'] = duplicate_results
            
            # Stage 3: Best Photo Selection
            print("\\nStage 3: Selecting best photos...")
            best_photos = self._select_best_from_clusters(duplicate_results['similar_clusters'])
            results['stages']['selection'] = {'best_photos': len(best_photos)}
            
            # Copy best photos to special directory
            self._organize_best_photos(best_photos)
        else:
            print("\\nSkipping clustering - not enough sharp images")
            results['stages']['duplicates'] = {'skipped': 'insufficient_images'}
            results['stages']['selection'] = {'skipped': 'insufficient_images'}
        
        print("\\n✅ Processing complete!")
        results['success'] = True
        return results
    
    def process_visa_photo(self, input_path: str, output_path: Optional[str] = None) -> Dict:
        """Process a single photo for visa requirements."""
        if not output_path:
            input_name = Path(input_path).stem
            output_path = os.path.join(self.config.output_dir, f"{input_name}_visa.jpg")
        
        print(f"Processing visa photo: {input_path}")
        result = self.visa_processor.process_visa_photo(input_path, output_path)
        
        if result['success']:
            print(f"✅ Visa photo created: {output_path}")
        else:
            print(f"❌ Visa photo processing failed: {result['error']}")
        
        return result
    
    def analyze_photo_quality(self, image_paths: List[str]) -> Dict:
        """Comprehensive quality analysis of photos."""
        results = {
            'sharpness': self.sharpness_analyzer.batch_analyze(image_paths),
            'duplicates': self.duplicate_detector.find_comprehensive_duplicates(image_paths),
            'faces': {}
        }
        
        # Analyze faces in each image
        for path in image_paths:
            results['faces'][path] = self.face_detector.analyze_face_quality(path)
        
        return results
    
    def _organize_by_sharpness(self, sharpness_results: Dict):
        """Organize photos by sharpness into directories."""
        for image_path, result in sharpness_results.items():
            filename = os.path.basename(image_path)
            
            if result.get('overall_is_sharp', False):
                dest = os.path.join(self.config.good_dir, filename)
            else:
                dest = os.path.join(self.config.bad_dir, filename)
            
            try:
                shutil.copy2(image_path, dest)
            except Exception as e:
                print(f"Warning: Could not copy {image_path}: {e}")
    
    def _select_best_from_clusters(self, clusters: Dict) -> List[str]:
        """Select best photos from similarity clusters."""
        best_photos = []
        
        for cluster_id, cluster_paths in clusters.items():
            if cluster_id == -1:  # Skip noise
                continue
            
            # Analyze quality for all photos in cluster
            quality_scores = []
            for path in cluster_paths:
                face_result = self.face_detector.analyze_face_quality(path)
                sharpness_result = self.sharpness_analyzer.analyze_comprehensive(path)
                
                # Combined quality score
                face_score = face_result.get('quality_score', 0)
                sharpness_score = sharpness_result.get('confidence', 0)
                combined_score = (face_score * 0.6) + (sharpness_score * 0.4)
                
                quality_scores.append((path, combined_score))
            
            # Sort by quality and select top photos
            quality_scores.sort(key=lambda x: x[1], reverse=True)
            num_to_select = min(self.config.processing.num_best_photos, len(quality_scores))
            
            for path, score in quality_scores[:num_to_select]:
                best_photos.append(path)
        
        return best_photos
    
    def _organize_best_photos(self, best_photos: List[str]):
        """Copy best photos to a special directory."""
        best_dir = os.path.join(self.config.good_dir, "best")
        os.makedirs(best_dir, exist_ok=True)
        
        for photo_path in best_photos:
            filename = os.path.basename(photo_path)
            dest_path = os.path.join(best_dir, filename)
            
            try:
                shutil.copy2(photo_path, dest_path)
            except Exception as e:
                print(f"Warning: Could not copy best photo {photo_path}: {e}")
    
    def get_processing_stats(self) -> Dict:
        """Get statistics about processed photos."""
        stats = {
            'directories': {
                'input': self.config.input_dir,
                'good': self.config.good_dir,
                'bad': self.config.bad_dir,
                'output': self.config.output_dir
            },
            'counts': {}
        }
        
        # Count files in each directory
        for name, directory in stats['directories'].items():
            if os.path.exists(directory):
                image_count = len(get_image_paths(directory))
                stats['counts'][name] = image_count
            else:
                stats['counts'][name] = 0
        
        # Count best photos if directory exists
        best_dir = os.path.join(self.config.good_dir, "best")
        if os.path.exists(best_dir):
            stats['counts']['best'] = len(get_image_paths(best_dir))
        else:
            stats['counts']['best'] = 0
        
        return stats