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
from ..processors.enhancement.auto_enhancer import AutoEnhancer
from ..utils.image_utils import get_image_paths
from ..utils.filename_generator import SmartFilenameGenerator


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
        self.auto_enhancer = AutoEnhancer(self.config.__dict__)
        self.filename_generator = SmartFilenameGenerator(self.config.__dict__)

    def process_photos_pipeline(self, input_dir: Optional[str] = None) -> Dict:
        """Run complete photo processing pipeline."""
        input_dir = input_dir or self.config.input_dir

        if not os.path.exists(input_dir):
            return {"success": False, "error": f"Input directory does not exist: {input_dir}"}

        print(f"Processing photos from: {input_dir}")

        # Get all image paths
        image_paths = get_image_paths(input_dir)
        if not image_paths:
            return {"success": False, "error": "No images found in input directory"}

        print(f"Found {len(image_paths)} images")

        results = {"input_dir": input_dir, "total_images": len(image_paths), "stages": {}}

        # Stage 1: Sharpness Analysis
        print("\\nStage 1: Analyzing sharpness...")
        sharpness_results = self.sharpness_analyzer.batch_analyze(image_paths)
        sharp_images = [
            path
            for path, result in sharpness_results.items()
            if result.get("overall_is_sharp", False)
        ]

        # Move images to appropriate directories
        self._organize_by_sharpness(sharpness_results)

        results["stages"]["sharpness"] = {
            "processed": len(sharpness_results),
            "sharp": len(sharp_images),
            "blurry": len(image_paths) - len(sharp_images),
        }

        print(f"Sharp images: {len(sharp_images)}, Blurry: {len(image_paths) - len(sharp_images)}")

        # Stage 2: Duplicate Detection
        if len(sharp_images) >= self.config.processing.min_photos_to_cluster:
            print("\\nStage 2: Detecting duplicates and clustering...")
            duplicate_results = self.duplicate_detector.find_comprehensive_duplicates(sharp_images)
            results["stages"]["duplicates"] = duplicate_results

            # Stage 3: Best Photo Selection
            print("\\nStage 3: Selecting best photos...")
            best_photos = self._select_best_from_clusters(duplicate_results["similar_clusters"])
            results["stages"]["selection"] = {"best_photos": len(best_photos)}

            # Copy best photos to special directory
            self._organize_best_photos(best_photos)
        else:
            print("\\nSkipping clustering - not enough sharp images")
            results["stages"]["duplicates"] = {"skipped": "insufficient_images"}
            results["stages"]["selection"] = {"skipped": "insufficient_images"}

        print("\\n✅ Processing complete!")
        results["success"] = True
        return results

    def process_visa_photo(self, input_path: str, output_path: Optional[str] = None) -> Dict:
        """Process a single photo for visa requirements."""
        if not output_path:
            input_name = Path(input_path).stem
            output_path = os.path.join(self.config.output_dir, f"{input_name}_visa.jpg")

        print(f"Processing visa photo: {input_path}")
        result = self.visa_processor.process_visa_photo(input_path, output_path)

        if result["success"]:
            print(f"✅ Visa photo created: {output_path}")
        else:
            print(f"❌ Visa photo processing failed: {result['error']}")

        return result

    def analyze_photo_quality(self, image_paths: List[str]) -> Dict:
        """Comprehensive quality analysis of photos."""
        results = {
            "sharpness": self.sharpness_analyzer.batch_analyze(image_paths),
            "duplicates": self.duplicate_detector.find_comprehensive_duplicates(image_paths),
            "faces": {},
        }

        # Analyze faces in each image
        for path in image_paths:
            results["faces"][path] = self.face_detector.analyze_face_quality(path)

        return results

    def _organize_by_sharpness(self, sharpness_results: Dict):
        """Organize photos by sharpness into directories."""
        for image_path, result in sharpness_results.items():
            filename = os.path.basename(image_path)

            if result.get("overall_is_sharp", False):
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
                face_score = face_result.get("quality_score", 0)
                sharpness_score = sharpness_result.get("confidence", 0)
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
            "directories": {
                "input": self.config.input_dir,
                "good": self.config.good_dir,
                "bad": self.config.bad_dir,
                "output": self.config.output_dir,
            },
            "counts": {},
        }

        # Count files in each directory
        for name, directory in stats["directories"].items():
            if os.path.exists(directory):
                image_count = len(get_image_paths(directory))
                stats["counts"][name] = image_count
            else:
                stats["counts"][name] = 0

        # Count best photos if directory exists
        best_dir = os.path.join(self.config.good_dir, "best")
        if os.path.exists(best_dir):
            stats["counts"]["best"] = len(get_image_paths(best_dir))
        else:
            stats["counts"]["best"] = 0

        return stats
    
    def process_sports_photos(self, input_dir: Optional[str] = None, output_dir: Optional[str] = None) -> Dict:
        """
        Complete sports photo processing pipeline with enhancement and smart naming.
        
        This method implements your specific requirements:
        1. Filter bad photos (blur, poor exposure, out of focus)
        2. Select best photos from similar groups
        3. Automatically enhance good photos
        4. Generate smart filenames with EXIF timestamps
        5. Organize output with before/after structure
        """
        input_dir = input_dir or self.config.input_dir
        output_dir = output_dir or os.path.join(input_dir, 'output')
        
        if not os.path.exists(input_dir):
            return {"success": False, "error": f"Input directory does not exist: {input_dir}"}
        
        print(f"🏆 Processing sports photos from: {input_dir}")
        print(f"📁 Output will be saved to: {output_dir}")
        
        # Create organized output structure
        output_structure = self.filename_generator.generate_output_structure(output_dir)
        
        # Get all image paths
        image_paths = get_image_paths(input_dir)
        if not image_paths:
            return {"success": False, "error": "No images found in input directory"}
        
        print(f"📸 Found {len(image_paths)} photos to process")
        
        results = {
            "success": True,
            "input_dir": input_dir,
            "output_dir": output_dir,
            "total_images": len(image_paths),
            "processed_images": [],
            "discarded_images": [],
            "enhanced_images": [],
            "statistics": {}
        }
        
        # Stage 1: Filter bad photos
        print(f"\\n🔍 Stage 1: Filtering bad photos...")
        good_photos = []
        discarded_photos = []
        
        for i, image_path in enumerate(image_paths, 1):
            try:
                print(f"  Analyzing {i}/{len(image_paths)}: {os.path.basename(image_path)}")
                
                # Check sharpness with error handling
                try:
                    sharpness_result = self.sharpness_analyzer.analyze_comprehensive(image_path)
                    is_sharp = sharpness_result.get('overall_is_sharp', False)
                except Exception as e:
                    print(f"    Warning: Sharpness analysis failed: {e}")
                    sharpness_result = {'overall_is_sharp': False, 'confidence': 0.0}
                    is_sharp = False
                
                # Check motion blur with error handling
                try:
                    motion_blur_result = self.auto_enhancer.detect_motion_blur(image_path)
                    has_motion_blur = motion_blur_result.get('has_motion_blur', False)
                except Exception as e:
                    print(f"    Warning: Motion blur detection failed: {e}")
                    motion_blur_result = {'has_motion_blur': False}
                    has_motion_blur = False
                
                # Check exposure with error handling  
                try:
                    exposure_result = self.auto_enhancer.detect_poor_exposure(image_path)
                    has_poor_exposure = exposure_result.get('has_poor_exposure', False)
                except Exception as e:
                    print(f"    Warning: Exposure analysis failed: {e}")
                    exposure_result = {'has_poor_exposure': False}
                    has_poor_exposure = False
                    
                # Memory cleanup every few images
                if i % 3 == 0:
                    import gc
                    gc.collect()
            
                # Decision logic
                if is_sharp and not has_motion_blur and not has_poor_exposure:
                    good_photos.append({
                        'path': image_path,
                        'sharpness_score': sharpness_result.get('confidence', 0),
                        'quality_metrics': {
                            'sharpness': sharpness_result,
                            'motion_blur': motion_blur_result,
                            'exposure': exposure_result
                        }
                    })
                else:
                    discarded_photos.append({
                        'path': image_path,
                        'reasons': [],
                        'quality_metrics': {
                            'sharpness': sharpness_result,
                            'motion_blur': motion_blur_result,
                            'exposure': exposure_result
                        }
                    })
                    
                    # Record discard reasons
                    reasons = discarded_photos[-1]['reasons']
                    if not is_sharp:
                        reasons.append('not_sharp')
                    if has_motion_blur:
                        reasons.append('motion_blur')
                    if has_poor_exposure:
                        reasons.append('poor_exposure')
                        
            except Exception as e:
                print(f"  Critical error processing {image_path}: {e}")
                # Add to discarded photos with error reason
                discarded_photos.append({
                    'path': image_path,
                    'reasons': ['processing_error'],
                    'error': str(e),
                    'quality_metrics': {}
                })
        
        print(f"✅ Stage 1 complete: {len(good_photos)} good photos, {len(discarded_photos)} discarded")
        
        if not good_photos:
            return {"success": False, "error": "No good quality photos found"}
        
        # Stage 2: Find similar photos and select best
        print(f"\\n🎯 Stage 2: Selecting best photos from similar groups...")
        
        good_photo_paths = [photo['path'] for photo in good_photos]
        duplicate_results = self.duplicate_detector.find_comprehensive_duplicates(good_photo_paths)
        clusters = duplicate_results.get('clusters', {})
        
        # Select best photos from each cluster
        selected_photos = []
        for cluster_id, cluster_paths in clusters.items():
            if cluster_id == -1:  # Unclustered (unique) photos - keep all
                for path in cluster_paths:
                    photo_data = next(p for p in good_photos if p['path'] == path)
                    selected_photos.append(photo_data)
            else:
                # Find best photo in cluster
                cluster_photos = [p for p in good_photos if p['path'] in cluster_paths]
                best_photo = max(cluster_photos, key=lambda x: x['sharpness_score'])
                selected_photos.append(best_photo)
        
        print(f"✅ Stage 2 complete: Selected {len(selected_photos)} best photos from {len(good_photos)} good photos")
        
        # Stage 3: Enhance selected photos
        print(f"\\n✨ Stage 3: Enhancing selected photos...")
        
        enhanced_results = []
        filename_mapping = {}
        
        for i, photo in enumerate(selected_photos, 1):
            image_path = photo['path']
            print(f"  Enhancing {i}/{len(selected_photos)}: {os.path.basename(image_path)}")
            
            # Generate smart filename
            new_filename = self.filename_generator.generate_filename(
                image_path, 
                'best', 
                photo['sharpness_score']
            )
            
            # Enhance photo
            enhanced_path = os.path.join(output_structure['enhanced_photos'], new_filename)
            enhancement_result = self.auto_enhancer.enhance_photo(image_path, enhanced_path)
            
            if enhancement_result['success']:
                enhanced_results.append({
                    'original_path': image_path,
                    'enhanced_path': enhanced_path,
                    'new_filename': new_filename,
                    'quality_score': photo['sharpness_score'],
                    'enhancements': enhancement_result['enhancements'],
                    'success': True
                })
                
                # Also copy original to best_photos directory for comparison
                best_photo_path = os.path.join(output_structure['best_photos'], new_filename)
                shutil.copy2(image_path, best_photo_path)
                
            else:
                enhanced_results.append({
                    'original_path': image_path,
                    'error': enhancement_result['error'],
                    'success': False
                })
        
        print(f"✅ Stage 3 complete: Enhanced {sum(1 for r in enhanced_results if r['success'])} photos")
        
        # Stage 4: Handle discarded photos
        print(f"\\n🗑️ Stage 4: Organizing discarded photos...")
        
        for photo_data in discarded_photos:
            image_path = photo_data['path']
            filename = os.path.basename(image_path)
            
            # Create reason-based subdirectories
            main_reason = photo_data['reasons'][0] if photo_data['reasons'] else 'unknown'
            reason_dir = os.path.join(output_structure['discarded_photos'], main_reason)
            os.makedirs(reason_dir, exist_ok=True)
            
            dest_path = os.path.join(reason_dir, filename)
            shutil.copy2(image_path, dest_path)
        
        # Stage 5: Generate summary
        print(f"\\n📊 Stage 5: Generating processing summary...")
        
        summary_path = self.filename_generator.create_processing_summary(enhanced_results, output_dir)
        
        # Update results
        results.update({
            "processed_images": enhanced_results,
            "discarded_images": discarded_photos,
            "enhanced_images": [r for r in enhanced_results if r['success']],
            "statistics": {
                "total_input": len(image_paths),
                "good_quality": len(good_photos),
                "selected_best": len(selected_photos),
                "successfully_enhanced": sum(1 for r in enhanced_results if r['success']),
                "discarded": len(discarded_photos),
                "success_rate": len(selected_photos) / len(image_paths) * 100
            },
            "output_structure": output_structure,
            "summary_file": summary_path
        })
        
        print(f"\\n🎉 Sports photo processing complete!")
        print(f"📊 Results: {results['statistics']['successfully_enhanced']}/{results['statistics']['total_input']} photos processed successfully ({results['statistics']['success_rate']:.1f}% success rate)")
        print(f"📁 Enhanced photos: {output_structure['enhanced_photos']}")
        print(f"📁 Original best photos: {output_structure['best_photos']}")
        print(f"📄 Summary: {summary_path}")
        
        return results
