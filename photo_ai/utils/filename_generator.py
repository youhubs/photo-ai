"""Smart filename generation for processed photos."""

import os
from datetime import datetime
from typing import Optional, Dict
from PIL import Image
from .time_utils import get_capture_time


class SmartFilenameGenerator:
    """Generate intelligent filenames for processed photos."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.use_exif_time = self.config.get('use_exif_time', True)
        self.fallback_to_file_time = self.config.get('fallback_to_file_time', True)
        self.add_quality_suffix = self.config.get('add_quality_suffix', True)
    
    def generate_filename(
        self, 
        original_path: str, 
        processing_type: str = 'best',
        quality_score: Optional[float] = None,
        sequence_number: Optional[int] = None
    ) -> str:
        """
        Generate smart filename based on EXIF timestamp and processing info.
        
        Args:
            original_path: Path to original image
            processing_type: Type of processing ('best', 'enhanced', 'selected')
            quality_score: Optional quality score (0-1)
            sequence_number: Optional sequence number for batch processing
            
        Returns:
            Generated filename (without path)
        """
        # Get timestamp
        timestamp = self._get_timestamp(original_path)
        
        # Format timestamp
        if timestamp:
            time_str = timestamp.strftime("%Y%m%d_%H%M%S")
        else:
            # Fallback to current time if no timestamp available
            time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Build filename components
        components = [time_str]
        
        # Add processing type
        components.append(processing_type)
        
        # Add quality score if provided
        if quality_score is not None and self.add_quality_suffix:
            quality_str = f"q{int(quality_score * 100):02d}"
            components.append(quality_str)
        
        # Add sequence number if provided
        if sequence_number is not None:
            components.append(f"seq{sequence_number:03d}")
        
        # Join components and add extension
        filename = "_".join(components) + ".jpg"
        
        return filename
    
    def generate_batch_filenames(
        self, 
        photo_results: list,
        processing_type: str = 'best'
    ) -> Dict[str, str]:
        """
        Generate filenames for a batch of processed photos.
        
        Args:
            photo_results: List of photo processing results
            processing_type: Type of processing applied
            
        Returns:
            Dict mapping original_path -> new_filename
        """
        filename_mapping = {}
        
        # Group by timestamp to handle duplicates
        timestamp_groups = {}
        
        for i, result in enumerate(photo_results):
            original_path = result.get('original_path')
            if not original_path:
                continue
                
            timestamp = self._get_timestamp(original_path)
            time_key = timestamp.strftime("%Y%m%d_%H%M%S") if timestamp else f"unknown_{i}"
            
            if time_key not in timestamp_groups:
                timestamp_groups[time_key] = []
            
            timestamp_groups[time_key].append({
                'original_path': original_path,
                'quality_score': result.get('quality_score'),
                'result': result
            })
        
        # Generate filenames for each group
        for time_key, group in timestamp_groups.items():
            if len(group) == 1:
                # Single photo for this timestamp
                photo = group[0]
                filename = self.generate_filename(
                    photo['original_path'],
                    processing_type,
                    photo['quality_score']
                )
                filename_mapping[photo['original_path']] = filename
            else:
                # Multiple photos for same timestamp - add sequence numbers
                # Sort by quality score (highest first)
                sorted_group = sorted(
                    group, 
                    key=lambda x: x['quality_score'] or 0, 
                    reverse=True
                )
                
                for seq_num, photo in enumerate(sorted_group, 1):
                    filename = self.generate_filename(
                        photo['original_path'],
                        processing_type,
                        photo['quality_score'],
                        seq_num
                    )
                    filename_mapping[photo['original_path']] = filename
        
        return filename_mapping
    
    def _get_timestamp(self, image_path: str) -> Optional[datetime]:
        """Get timestamp from image (EXIF or file time)."""
        if self.use_exif_time:
            # Try to get EXIF timestamp first
            exif_time = get_capture_time(image_path)
            if exif_time:
                return exif_time
        
        if self.fallback_to_file_time:
            # Fall back to file modification time
            try:
                timestamp = os.path.getmtime(image_path)
                return datetime.fromtimestamp(timestamp)
            except Exception:
                pass
        
        return None
    
    def generate_output_structure(self, base_output_dir: str) -> Dict[str, str]:
        """
        Generate organized output directory structure.
        
        Returns:
            Dict with directory paths for different output types
        """
        structure = {
            'best_photos': os.path.join(base_output_dir, 'best_photos'),
            'enhanced_photos': os.path.join(base_output_dir, 'enhanced_photos'),
            'discarded_photos': os.path.join(base_output_dir, 'discarded_photos'),
            'processing_log': os.path.join(base_output_dir, 'processing_log.txt'),
            'statistics': os.path.join(base_output_dir, 'statistics.json')
        }
        
        # Create directories if they don't exist
        for dir_path in structure.values():
            if dir_path.endswith('.txt') or dir_path.endswith('.json'):
                # Skip files, only create directories
                continue
            os.makedirs(dir_path, exist_ok=True)
        
        return structure
    
    def create_processing_summary(
        self, 
        results: list, 
        output_dir: str
    ) -> str:
        """
        Create a summary file of the processing results.
        
        Returns:
            Path to created summary file
        """
        summary_path = os.path.join(output_dir, 'processing_summary.txt')
        
        total_processed = len(results)
        successful = sum(1 for r in results if r.get('success', False))
        enhanced = sum(1 for r in results if r.get('enhanced', False))
        discarded = sum(1 for r in results if r.get('discarded', False))
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("Photo AI - Sports Photo Processing Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Statistics:\n")
            f.write(f"  Total Photos Processed: {total_processed}\n")
            f.write(f"  Successfully Processed: {successful}\n")
            f.write(f"  Enhanced Photos: {enhanced}\n")
            f.write(f"  Discarded Photos: {discarded}\n")
            f.write(f"  Success Rate: {(successful/total_processed*100):.1f}%\n\n")
            
            f.write("Enhancements Applied:\n")
            enhancement_counts = {}
            for result in results:
                enhancements = result.get('enhancements', [])
                for enhancement in enhancements:
                    enhancement_counts[enhancement] = enhancement_counts.get(enhancement, 0) + 1
            
            for enhancement, count in sorted(enhancement_counts.items()):
                f.write(f"  {enhancement}: {count} photos\n")
            
            f.write("\nDetailed Results:\n")
            f.write("-" * 30 + "\n")
            
            for i, result in enumerate(results, 1):
                original = result.get('original_path', 'Unknown')
                new_filename = result.get('new_filename', 'Unknown')
                status = "SUCCESS" if result.get('success') else "FAILED"
                
                f.write(f"\n{i:3d}. {os.path.basename(original)}\n")
                f.write(f"     -> {new_filename} [{status}]\n")
                
                if result.get('quality_score'):
                    f.write(f"     Quality: {result['quality_score']:.2f}\n")
                
                if result.get('enhancements'):
                    f.write(f"     Enhancements: {', '.join(result['enhancements'])}\n")
        
        return summary_path