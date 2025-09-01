"""Main photo processing orchestrator with GUI logging support."""

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
from ..processors.sports.player_grouping import PlayerGroupingProcessor
from ..utils.image_utils import get_image_paths
from ..utils.filename_generator import SmartFilenameGenerator


class PhotoProcessor:
    """Main photo processing orchestrator with logging support."""

    def __init__(self, config: Optional[Config] = None, logger: Optional[callable] = None):
        self.config = config or Config.from_env()
        self.config.create_directories()

        # Initialize processors
        self.sharpness_analyzer = SharpnessAnalyzer(self.config)
        self.duplicate_detector = DuplicateDetector(self.config)
        self.face_detector = FaceDetector(self.config)
        self.visa_processor = VisaPhotoProcessor(self.config)
        self.auto_enhancer = AutoEnhancer(self.config.__dict__)
        self.filename_generator = SmartFilenameGenerator(self.config.__dict__)
        self.player_grouping = PlayerGroupingProcessor(self.config)

        # Logger callback (accepts str)
        self.logger = logger or print

    def log(self, message: str):
        """Send message to console and GUI logger if available."""
        self.logger(message)

    def process_photos_pipeline(self, input_dir: Optional[str] = None) -> Dict:
        """Run complete photo processing pipeline."""
        input_dir = input_dir or self.config.input_dir

        if not os.path.exists(input_dir):
            self.log(f"❌ Input directory does not exist: {input_dir}")
            return {"success": False, "error": f"Input directory does not exist: {input_dir}"}

        self.log(f"📁 Processing photos from: {input_dir}")

        # Get all image paths
        image_paths = get_image_paths(input_dir)
        if not image_paths:
            self.log("⚠️ No images found in input directory")
            return {"success": False, "error": "No images found in input directory"}

        self.log(f"📸 Found {len(image_paths)} images")

        results = {"input_dir": input_dir, "total_images": len(image_paths), "stages": {}}

        # Stage 1: Sharpness Analysis
        self.log("\n🔍 Stage 1: Analyzing sharpness...")
        sharpness_results = self.sharpness_analyzer.batch_analyze(image_paths)
        sharp_images = [
            path
            for path, result in sharpness_results.items()
            if result.get("overall_is_sharp", False)
        ]

        # Organize by sharpness
        self._organize_by_sharpness(sharpness_results)
        results["stages"]["sharpness"] = {
            "processed": len(sharpness_results),
            "sharp": len(sharp_images),
            "blurry": len(image_paths) - len(sharp_images),
        }
        self.log(
            f"✅ Sharp images: {len(sharp_images)}, Blurry: {len(image_paths) - len(sharp_images)}"
        )

        # Stage 2: Duplicate Detection
        if len(sharp_images) >= self.config.processing.min_photos_to_cluster:
            self.log("\n🔗 Stage 2: Detecting duplicates and clustering...")
            duplicate_results = self.duplicate_detector.find_comprehensive_duplicates(sharp_images)
            results["stages"]["duplicates"] = duplicate_results

            # Stage 3: Best Photo Selection
            self.log("🎯 Stage 3: Selecting best photos...")
            best_photos = self._select_best_from_clusters(
                duplicate_results.get("similar_clusters", {})
            )
            results["stages"]["selection"] = {"best_photos": len(best_photos)}
            self._organize_best_photos(best_photos)
        else:
            self.log("⚠️ Skipping clustering - not enough sharp images")
            results["stages"]["duplicates"] = {"skipped": "insufficient_images"}
            results["stages"]["selection"] = {"skipped": "insufficient_images"}

        self.log("🎉 Processing complete!")
        results["success"] = True
        return results

    def process_visa_photo(self, input_path: str, output_path: Optional[str] = None) -> Dict:
        """Process a single photo for visa requirements."""
        if not output_path:
            input_name = Path(input_path).stem
            output_path = os.path.join(self.config.output_dir, f"{input_name}_visa.jpg")

        self.log(f"📄 Processing visa photo: {input_path}")
        result = self.visa_processor.process_visa_photo(input_path, output_path)

        if result["success"]:
            self.log(f"✅ Visa photo created: {output_path}")
        else:
            self.log(f"❌ Visa photo processing failed: {result['error']}")

        return result

    def analyze_photo_quality(self, image_paths: List[str]) -> Dict:
        """Comprehensive quality analysis of photos."""
        self.log("🔍 Analyzing photo quality...")
        results = {
            "sharpness": self.sharpness_analyzer.batch_analyze(image_paths),
            "duplicates": self.duplicate_detector.find_comprehensive_duplicates(image_paths),
            "faces": {},
        }

        # Analyze faces in each image
        for path in image_paths:
            results["faces"][path] = self.face_detector.analyze_face_quality(path)

        self.log("✅ Analysis complete")
        return results

    def _organize_by_sharpness(self, sharpness_results: Dict):
        """Organize photos by sharpness into directories."""
        for image_path, result in sharpness_results.items():
            filename = os.path.basename(image_path)
            dest = os.path.join(
                (
                    self.config.good_dir
                    if result.get("overall_is_sharp", False)
                    else self.config.bad_dir
                ),
                filename,
            )
            try:
                shutil.copy2(image_path, dest)
            except Exception as e:
                self.log(f"⚠️ Could not copy {image_path}: {e}")

    def _select_best_from_clusters(self, clusters: Dict) -> List[str]:
        """Select best photos from similarity clusters."""
        best_photos = []

        for cluster_id, cluster_paths in clusters.items():
            if cluster_id == -1:  # Skip noise
                continue

            quality_scores = []
            for path in cluster_paths:
                face_result = self.face_detector.analyze_face_quality(path)
                sharpness_result = self.sharpness_analyzer.analyze_comprehensive(path)
                face_score = face_result.get("quality_score", 0)
                sharpness_score = sharpness_result.get("confidence", 0)
                combined_score = (face_score * 0.6) + (sharpness_score * 0.4)
                quality_scores.append((path, combined_score))

            quality_scores.sort(key=lambda x: x[1], reverse=True)
            num_to_select = min(self.config.processing.num_best_photos, len(quality_scores))
            best_photos.extend([path for path, _ in quality_scores[:num_to_select]])

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
                self.log(f"⚠️ Could not copy best photo {photo_path}: {e}")

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

        for name, directory in stats["directories"].items():
            stats["counts"][name] = (
                len(get_image_paths(directory)) if os.path.exists(directory) else 0
            )

        best_dir = os.path.join(self.config.good_dir, "best")
        stats["counts"]["best"] = len(get_image_paths(best_dir)) if os.path.exists(best_dir) else 0

        return stats

    def process_sports_photos(
        self, input_dir: Optional[str] = None, output_dir: Optional[str] = None
    ) -> Dict:
        """Complete sports photo pipeline with filtering, enhancement, and smart filenames."""
        input_dir = input_dir or self.config.input_dir
        output_dir = output_dir or os.path.join(input_dir, "output")

        if not os.path.exists(input_dir):
            self.log(f"❌ Input directory does not exist: {input_dir}")
            return {"success": False, "error": f"Input directory does not exist: {input_dir}"}

        self.log(f"🏆 Processing sports photos from: {input_dir}")
        self.log(f"📁 Output will be saved to: {output_dir}")

        output_structure = self.filename_generator.generate_output_structure(output_dir)
        image_paths = get_image_paths(input_dir)

        if not image_paths:
            self.log("⚠️ No images found in input directory")
            return {"success": False, "error": "No images found in input directory"}

        results = {
            "success": True,
            "input_dir": input_dir,
            "output_dir": output_dir,
            "total_images": len(image_paths),
            "processed_images": [],
            "discarded_images": [],
            "enhanced_images": [],
            "statistics": {},
        }

        # Stage 1: Filter bad photos
        self.log("\n🔍 Stage 1: Filtering bad photos...")
        good_photos, discarded_photos = [], []

        for i, image_path in enumerate(image_paths, 1):
            self.log(f"  Analyzing {i}/{len(image_paths)}: {os.path.basename(image_path)}")
            try:
                sharpness_result = self.sharpness_analyzer.analyze_comprehensive(image_path)
                is_sharp = sharpness_result.get("overall_is_sharp", False)
                motion_blur_result = self.auto_enhancer.detect_motion_blur(image_path)
                exposure_result = self.auto_enhancer.detect_poor_exposure(image_path)
                has_motion_blur = motion_blur_result.get("has_motion_blur", False)
                has_poor_exposure = exposure_result.get("has_poor_exposure", False)

                if is_sharp and not has_motion_blur and not has_poor_exposure:
                    good_photos.append(
                        {
                            "path": image_path,
                            "sharpness_score": sharpness_result.get("confidence", 0),
                        }
                    )
                else:
                    discarded_photos.append(
                        {
                            "path": image_path,
                            "reasons": [],
                        }
                    )
                    reasons = discarded_photos[-1]["reasons"]
                    if not is_sharp:
                        reasons.append("not_sharp")
                    if has_motion_blur:
                        reasons.append("motion_blur")
                    if has_poor_exposure:
                        reasons.append("poor_exposure")

            except Exception as e:
                self.log(f"⚠️ Error processing {image_path}: {e}")
                discarded_photos.append({"path": image_path, "reasons": ["processing_error"]})

        self.log(
            f"✅ Stage 1 complete: {len(good_photos)} good photos, {len(discarded_photos)} discarded"
        )

        if not good_photos:
            return {"success": False, "error": "No good quality photos found"}

        # Stage 2: Select best photos
        self.log("\n🎯 Stage 2: Selecting best photos from similar groups...")
        good_photo_paths = [p["path"] for p in good_photos]
        duplicate_results = self.duplicate_detector.find_comprehensive_duplicates(good_photo_paths)
        clusters = duplicate_results.get("clusters", {})
        selected_photos = []

        for cluster_id, cluster_paths in clusters.items():
            if cluster_id == -1:
                for path in cluster_paths:
                    selected_photos.append(next(p for p in good_photos if p["path"] == path))
            else:
                cluster_photos = [p for p in good_photos if p["path"] in cluster_paths]
                best_photo = max(cluster_photos, key=lambda x: x["sharpness_score"])
                selected_photos.append(best_photo)

        self.log(f"✅ Stage 2 complete: Selected {len(selected_photos)} best photos")

        # Stage 3: Enhance selected photos
        self.log("\n✨ Stage 3: Enhancing selected photos...")
        enhanced_results = []

        for i, photo in enumerate(selected_photos, 1):
            image_path = photo["path"]
            new_filename = self.filename_generator.generate_filename(
                image_path, "best", photo["sharpness_score"]
            )
            enhanced_path = os.path.join(output_structure["enhanced_photos"], new_filename)
            enhancement_result = self.auto_enhancer.enhance_photo(image_path, enhanced_path)

            if enhancement_result["success"]:
                enhanced_results.append(
                    {
                        "original_path": image_path,
                        "enhanced_path": enhanced_path,
                        "new_filename": new_filename,
                        "quality_score": photo["sharpness_score"],
                        "enhancements": enhancement_result["enhancements"],
                        "success": True,
                    }
                )
                best_photo_path = os.path.join(output_structure["best_photos"], new_filename)
                shutil.copy2(image_path, best_photo_path)
            else:
                enhanced_results.append(
                    {
                        "original_path": image_path,
                        "error": enhancement_result["error"],
                        "success": False,
                    }
                )

        self.log(
            f"✅ Stage 3 complete: Enhanced {sum(1 for r in enhanced_results if r['success'])} photos"
        )

        # Stage 4: Handle discarded photos
        self.log("\n🗑️ Stage 4: Organizing discarded photos...")
        for photo_data in discarded_photos:
            image_path = photo_data["path"]
            filename = os.path.basename(image_path)
            main_reason = photo_data["reasons"][0] if photo_data["reasons"] else "unknown"
            reason_dir = os.path.join(output_structure["discarded_photos"], main_reason)
            os.makedirs(reason_dir, exist_ok=True)
            dest_path = os.path.join(reason_dir, filename)
            shutil.copy2(image_path, dest_path)

        # Stage 5: Generate summary
        self.log("\n📊 Stage 5: Generating processing summary...")
        summary_path = self.filename_generator.create_processing_summary(
            enhanced_results, output_dir
        )

        results.update(
            {
                "processed_images": enhanced_results,
                "discarded_images": discarded_photos,
                "enhanced_images": [r for r in enhanced_results if r["success"]],
                "statistics": {
                    "total_input": len(image_paths),
                    "good_quality": len(good_photos),
                    "selected_best": len(selected_photos),
                    "successfully_enhanced": sum(1 for r in enhanced_results if r["success"]),
                    "discarded": len(discarded_photos),
                    "success_rate": len(selected_photos) / len(image_paths) * 100,
                },
                "output_structure": output_structure,
                "summary_file": summary_path,
            }
        )

        self.log(f"\n🎉 Sports photo processing complete!")
        self.log(
            f"📊 {results['statistics']['successfully_enhanced']}/{results['statistics']['total_input']} photos processed successfully ({results['statistics']['success_rate']:.1f}% success rate)"
        )
        self.log(f"📁 Enhanced photos: {output_structure['enhanced_photos']}")
        self.log(f"📁 Original best photos: {output_structure['best_photos']}")
        self.log(f"📄 Summary: {summary_path}")

        return results

    def group_photos_by_players(
        self, input_source: Optional[str] = None, players_dir: Optional[str] = None
    ) -> Dict:
        """
        Group photos by detected players using reference player matching.

        Args:
            input_source: Directory containing sports photos to process, or list of photo paths
            players_dir: Directory containing reference player photos (e.g., 'players/Messi.jpg', 'players/Ronaldo.jpg')

        Returns:
            Dictionary with grouping results and statistics
        """
        # Handle both folder path and file list
        if isinstance(input_source, str) and os.path.isdir(input_source):
            image_paths = get_image_paths(input_source)
            input_dir = input_source
        elif isinstance(input_source, list):
            image_paths = input_source
            input_dir = os.path.dirname(input_source[0]) if input_source else self.config.input_dir
        else:
            input_dir = input_source or self.config.input_dir
            image_paths = get_image_paths(input_dir)

        if not image_paths:
            self.log("⚠️ No images found for player grouping")
            return {"success": False, "error": "No images found"}

        # Default players directory if not specified
        if players_dir is None:
            players_dir = os.path.join(input_dir, "players")
            if not os.path.exists(players_dir):
                # Try looking for players folder in current working directory
                players_dir = "players"

        if not os.path.exists(players_dir):
            self.log(f"❌ Players directory not found: {players_dir}")
            self.log(
                "💡 Please create a 'players/' directory with reference photos (e.g., Messi.jpg, Ronaldo.jpg)"
            )
            return {"success": False, "error": f"Players directory not found: {players_dir}"}

        if not os.path.isdir(players_dir):
            self.log(f"❌ Players path exists but is not a directory: {players_dir}")
            self.log("💡 Please ensure 'players/' is a directory, not a file")
            return {"success": False, "error": f"Players path is not a directory: {players_dir}"}

        self.log(
            f"👥 Grouping {len(image_paths)} photos by players using references from {players_dir}..."
        )

        try:
            result = self.player_grouping.group_photos_by_players(
                image_paths, players_dir, self.log
            )
            if result.get("success"):
                self.log(
                    f"✅ Player grouping completed! Recognized {result['recognized_players']} players"
                )
                if result.get("photos_with_multiple_players", 0) > 0:
                    self.log(
                        f"👥 Found {result['photos_with_multiple_players']} photos with multiple players"
                    )
                if result.get("unknown_photos", 0) > 0:
                    self.log(f"❓ {result['unknown_photos']} photos moved to 'unknown' folder")
            return result
        except Exception as e:
            error_msg = f"Player grouping failed: {str(e)}"
            self.log(f"❌ {error_msg}")
            return {"success": False, "error": error_msg}

    def set_face_match_threshold(self, threshold: float):
        """
        Set the face matching threshold for player recognition.

        Args:
            threshold: Threshold value (0.0 to 1.0, higher = stricter matching)
        """
        self.player_grouping.set_face_match_threshold(threshold)
        self.log(f"🎯 Face matching threshold set to {threshold}")

    def set_visual_similarity_threshold(self, threshold: float):
        """
        Set the visual similarity threshold for non-face photo matching.

        Args:
            threshold: Threshold value (0.0 to 1.0, higher = stricter matching)
        """
        self.player_grouping.set_visual_similarity_threshold(threshold)
        self.log(f"🎨 Visual similarity threshold set to {threshold}")

    def enable_non_face_matching(self, enabled: bool = True):
        """
        Enable or disable matching of photos without detectable faces (back views, etc.).

        Args:
            enabled: Whether to enable non-face matching
        """
        self.player_grouping.enable_disable_non_face_matching(enabled)
        status = "enabled" if enabled else "disabled"
        self.log(f"👤 Non-face matching {status}")

    def get_loaded_players(self) -> List[str]:
        """Get list of currently loaded reference players."""
        return self.player_grouping.get_reference_players()

    def get_visual_reference_players(self) -> List[str]:
        """Get list of players with visual feature references."""
        return self.player_grouping.get_visual_reference_players()

    def enable_jersey_number_matching(self, enabled: bool = True):
        """
        Enable or disable jersey number detection and matching.

        Args:
            enabled: Whether to enable jersey number matching
        """
        self.player_grouping.enable_jersey_number_matching = enabled
        status = "enabled" if enabled else "disabled"
        self.log(f"🔢 Jersey number matching {status}")

    def get_jersey_number_mappings(self) -> Dict[str, int]:
        """Get current player jersey number mappings."""
        return self.player_grouping.player_jersey_numbers.copy()

    def process_soccer_photos_complete(
        self, input_dir: str, players_dir: str = None, output_dir: str = None
    ) -> Dict:
        """
        Complete soccer photo processing workflow organized by players:
        1. Remove bad-quality photos (blurry, out-of-focus)
        2. Remove duplicate or near-duplicate photos
        3. Group photos by player using reference faces
        4. Select best photos per player (1-2 best quality)
        5. Save organized by player folders

        Args:
            input_dir: Directory containing photos to process
            players_dir: Directory with reference player photos (default: input_dir/players)
            output_dir: Output directory (default: input_dir/output)

        Returns:
            Dictionary with processing results and statistics
        """
        return self._process_soccer_photos_internal(input_dir, players_dir, output_dir)

    def process_photos_by_people_complete(
        self, input_dir: str, people_dir: str = None, output_dir: str = None
    ) -> Dict:
        """
        Legacy method name - use process_soccer_photos_complete instead.
        Complete soccer photo processing workflow organized by players.
        """
        # Map new parameter names to old ones
        players_dir = people_dir if people_dir else None
        return self.process_soccer_photos_complete(input_dir, players_dir, output_dir)

    def _process_soccer_photos_internal(
        self, input_dir: str, players_dir: str = None, output_dir: str = None
    ) -> Dict:
        """
        Internal implementation of the complete soccer photo processing workflow.
        """
        if players_dir is None:
            players_dir = os.path.join(input_dir, "players")

        if output_dir is None:
            output_dir = os.path.join(input_dir, "output")

        self.log("📸 Starting comprehensive photo processing...")
        self.log(f"📁 Input directory: {input_dir}")
        self.log(f"👥 Players directory: {players_dir}")
        self.log(f"📤 Output directory: {output_dir}")

        if not os.path.exists(input_dir):
            return {"success": False, "error": f"Input directory not found: {input_dir}"}

        if not os.path.exists(players_dir):
            return {"success": False, "error": f"Players directory not found: {players_dir}"}

        # Get all game photos
        image_paths = get_image_paths(input_dir)
        if not image_paths:
            return {"success": False, "error": "No images found in input directory"}

        total_input = len(image_paths)
        self.log(f"📸 Found {total_input} photos to process")

        results = {
            "success": True,
            "total_input_photos": total_input,
            "players_dir": players_dir,
            "output_dir": output_dir,
            "stages": {},
        }

        try:
            # Stage 1: Quality Filtering - Remove blurry/bad photos
            self.log("🔍 Stage 1: Filtering out poor quality photos...")
            sharpness_results = self.sharpness_analyzer.analyze_images(image_paths)

            sharp_photos = []
            blurry_count = 0
            for photo_path, analysis in sharpness_results.items():
                if analysis.get("overall_is_sharp", False):
                    sharp_photos.append(photo_path)
                else:
                    blurry_count += 1

            self.log(
                f"✅ Quality filtering: {len(sharp_photos)} sharp photos, {blurry_count} blurry removed"
            )
            results["stages"]["quality_filter"] = {
                "sharp_photos": len(sharp_photos),
                "blurry_removed": blurry_count,
            }

            if not sharp_photos:
                return {"success": False, "error": "No sharp photos found after quality filtering"}

            # Stage 2: Duplicate Detection - Remove near-duplicates
            self.log("🔍 Stage 2: Detecting and removing duplicate photos...")
            duplicate_results = self.duplicate_detector.find_duplicates(sharp_photos)

            # Keep only unique photos (first occurrence of each duplicate group)
            unique_photos = []
            duplicates_removed = 0

            processed_groups = set()
            for photo_path in sharp_photos:
                # Check if this photo is part of a duplicate group we haven't processed
                photo_group = None
                for group in duplicate_results.get("duplicate_groups", []):
                    if photo_path in group and id(group) not in processed_groups:
                        photo_group = group
                        break

                if photo_group:
                    # Take the first photo from the group (could be enhanced to pick best quality)
                    unique_photos.append(photo_group[0])
                    duplicates_removed += len(photo_group) - 1
                    processed_groups.add(id(photo_group))
                elif photo_path not in [
                    p for group in duplicate_results.get("duplicate_groups", []) for p in group
                ]:
                    # Photo is not part of any duplicate group
                    unique_photos.append(photo_path)

            self.log(
                f"✅ Duplicate removal: {len(unique_photos)} unique photos, {duplicates_removed} duplicates removed"
            )
            results["stages"]["duplicate_removal"] = {
                "unique_photos": len(unique_photos),
                "duplicates_removed": duplicates_removed,
            }

            if not unique_photos:
                return {"success": False, "error": "No photos left after duplicate removal"}

            # Stage 3: Player Grouping
            self.log("🔍 Stage 3: Grouping photos by detected players...")
            grouping_results = self.group_photos_by_players(unique_photos, players_dir)

            if not grouping_results["success"]:
                return {
                    "success": False,
                    "error": f"Player grouping failed: {grouping_results['error']}",
                }

            self.log(
                f"✅ Player grouping: {grouping_results['recognized_players']} players recognized"
            )
            results["stages"]["player_grouping"] = {
                "recognized_players": grouping_results["recognized_players"],
                "unknown_photos": grouping_results["unknown_photos"],
                "group_stats": grouping_results.get("group_stats", {}),
            }

            # Stage 4: Best Photo Selection per Player
            self.log("🔍 Stage 4: Selecting best photos per player...")
            os.makedirs(output_dir, exist_ok=True)

            final_stats = {}
            total_selected = 0

            # Process each player's photos
            for player_name, stats in grouping_results.get("group_stats", {}).items():
                player_output_dir = os.path.join(output_dir, player_name)
                os.makedirs(person_output_dir, exist_ok=True)

                # Get all photos for this player from the grouping results
                player_photos = []
                player_source_dir = stats.get("folder_path")
                if player_source_dir and os.path.exists(player_source_dir):
                    for filename in os.listdir(player_source_dir):
                        if filename.lower().endswith(
                            (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
                        ):
                            player_photos.append(os.path.join(player_source_dir, filename))

                if not player_photos:
                    continue

                # Select best photos (analyze sharpness and pick top 2)
                best_photos = self._select_best_photos_for_player(player_photos, max_photos=2)

                # Copy selected photos to final output
                copied_count = 0
                for src_path in best_photos:
                    try:
                        filename = os.path.basename(src_path)
                        dst_path = os.path.join(player_output_dir, filename)
                        shutil.copy2(src_path, dst_path)
                        copied_count += 1
                    except Exception as e:
                        self.log(f"⚠️  Error copying {src_path}: {e}")

                final_stats[player_name] = {
                    "total_photos": len(player_photos),
                    "selected_photos": copied_count,
                    "output_dir": player_output_dir,
                }
                total_selected += copied_count

                self.log(
                    f"👤 {player_name}: Selected {copied_count} best photos from {len(player_photos)}"
                )

            # Handle unknown photos if any
            unknown_count = 0
            if grouping_results["unknown_photos"] > 0:
                unknown_dir = os.path.join(output_dir, "unknown")
                os.makedirs(unknown_dir, exist_ok=True)
                # Copy unknown photos would need access to the actual unknown photo paths
                # This would require modifying the grouping results to include the paths

            results["stages"]["best_selection"] = {
                "total_selected": total_selected,
                "player_stats": final_stats,
            }

            # Final summary
            self.log("✅ Soccer photo processing completed!")
            self.log(f"📊 Summary:")
            self.log(f"   Total input photos: {total_input}")
            self.log(f"   After quality filter: {len(sharp_photos)}")
            self.log(f"   After duplicate removal: {len(unique_photos)}")
            self.log(f"   Players recognized: {grouping_results['recognized_players']}")
            self.log(f"   Final photos selected: {total_selected}")
            self.log(f"📁 Output saved to: {output_dir}")

            results["final_summary"] = {
                "input_photos": total_input,
                "quality_filtered": len(sharp_photos),
                "unique_photos": len(unique_photos),
                "players_found": grouping_results["recognized_players"],
                "final_selected": total_selected,
                "output_directory": output_dir,
            }

            return results

        except Exception as e:
            error_msg = f"Error during soccer photo processing: {str(e)}"
            self.log(f"❌ {error_msg}")
            return {"success": False, "error": error_msg}

    def _select_best_photos_for_player(
        self, player_photos: List[str], max_photos: int = 2
    ) -> List[str]:
        """
        Select the best photos for a player based on quality metrics.

        Args:
            player_photos: List of photo paths for the player
            max_photos: Maximum number of photos to select

        Returns:
            List of paths to the best photos
        """
        if len(player_photos) <= max_photos:
            return player_photos

        # Analyze sharpness of all player photos
        photo_scores = []
        for photo_path in player_photos:
            try:
                analysis = self.sharpness_analyzer.analyze_image(photo_path)
                score = analysis.get("sharpness_score", 0.0)

                # Bonus for larger file sizes (often indicates better quality)
                try:
                    file_size = os.path.getsize(photo_path)
                    size_bonus = (
                        min(file_size / (1024 * 1024), 5.0) * 0.1
                    )  # Up to 0.5 bonus for files > 5MB
                    score += size_bonus
                except:
                    pass

                photo_scores.append((photo_path, score))
            except:
                # If analysis fails, give low score
                photo_scores.append((photo_path, 0.0))

        # Sort by score (highest first) and take top N
        photo_scores.sort(key=lambda x: x[1], reverse=True)
        return [photo_path for photo_path, score in photo_scores[:max_photos]]
