"""Player-based photo grouping for sports photos with reference player matching."""

import os
import shutil
from typing import Dict, List, Optional, Tuple
import numpy as np
import face_recognition
from pathlib import Path
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from ...core.config import Config
from ..face.detector import FaceDetector


class PlayerGroupingProcessor:
    """Groups sports photos by detected players using reference player photos."""

    def __init__(self, config: Config):
        self.config = config
        self.face_detector = FaceDetector(config)
        self.reference_players = {}  # Will store player_name -> encoding mapping
        self.reference_visual_features = (
            {}
        )  # Will store player_name -> visual features for non-face photos
        self.player_jersey_numbers = {}  # Will store player_name -> jersey_number mapping
        self.jersey_to_player = {}  # Will store jersey_number -> player_name mapping
        self.face_match_threshold = config.processing.face_match_threshold
        self.visual_similarity_threshold = config.processing.visual_similarity_threshold
        self.enable_non_face_matching = config.processing.enable_non_face_matching
        self.enable_jersey_number_matching = config.processing.enable_jersey_number_matching
        self.jersey_number_confidence_threshold = (
            config.processing.jersey_number_confidence_threshold
        )

        # Performance settings
        self.use_parallel_processing = config.processing.use_parallel_processing
        self.max_worker_threads = config.processing.max_worker_threads
        self.fast_mode = config.processing.fast_mode
        self.batch_size = config.processing.batch_size

    def _extract_enhanced_visual_features(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extract enhanced visual features optimized for sports player recognition.

        Args:
            image_path: Path to the image file

        Returns:
            Enhanced feature vector or None if extraction fails
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return None

            # Resize to standard size for consistent feature extraction
            image = cv2.resize(image, (224, 224))

            # Convert to different color spaces for richer features
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # 1. Enhanced color features (jersey colors, skin tone)
            # RGB histograms
            hist_r = cv2.calcHist([image_rgb], [0], None, [32], [0, 256])
            hist_g = cv2.calcHist([image_rgb], [1], None, [32], [0, 256])
            hist_b = cv2.calcHist([image_rgb], [2], None, [32], [0, 256])

            # HSV histograms (better for jersey color recognition)
            hist_h = cv2.calcHist([image_hsv], [0], None, [32], [0, 180])
            hist_s = cv2.calcHist([image_hsv], [1], None, [32], [0, 256])
            hist_v = cv2.calcHist([image_hsv], [2], None, [32], [0, 256])

            # 2. Body shape and posture features
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Edge detection for body outline
            edges = cv2.Canny(gray, 50, 150)
            edge_hist = np.histogram(edges, bins=16)[0]

            # 3. Spatial color distribution (where colors appear in the image)
            # Top half vs bottom half color differences (jersey vs shorts)
            h, w = image_rgb.shape[:2]
            top_half = image_rgb[: h // 2, :]
            bottom_half = image_rgb[h // 2 :, :]

            top_mean = np.mean(top_half.reshape(-1, 3), axis=0)
            bottom_mean = np.mean(bottom_half.reshape(-1, 3), axis=0)
            color_spatial = np.concatenate([top_mean, bottom_mean])

            # 4. Texture features optimized for fabric/clothing
            # LBP (Local Binary Patterns) for fabric texture
            from sklearn.feature_extraction import image as sk_image

            patches = sk_image.extract_patches_2d(gray, (16, 16), max_patches=50)
            texture_features = np.mean([np.std(patch) for patch in patches])

            # Combine all features
            color_features = np.concatenate(
                [
                    hist_r.flatten(),
                    hist_g.flatten(),
                    hist_b.flatten(),
                    hist_h.flatten(),
                    hist_s.flatten(),
                    hist_v.flatten(),
                ]
            )

            shape_features = edge_hist.flatten()
            spatial_features = color_spatial.flatten()

            # Combine all features
            features = np.concatenate(
                [color_features, shape_features, spatial_features, [texture_features]]
            )

            # Normalize features
            features = features / (np.linalg.norm(features) + 1e-8)

            return features

        except Exception as e:
            # Fallback to original method
            return self._extract_visual_features_basic(image_path)

    def _extract_visual_features(self, image_path: str) -> Optional[np.ndarray]:
        """Extract visual features with performance optimization."""
        if self.config.processing.fast_mode:
            return self._extract_visual_features_fast(image_path)
        else:
            enhanced = self._extract_enhanced_visual_features(image_path)
            return (
                enhanced
                if enhanced is not None
                else self._extract_visual_features_basic(image_path)
            )

    def _extract_visual_features_fast(self, image_path: str) -> Optional[np.ndarray]:
        """Fast visual feature extraction optimized for performance."""
        try:
            # Load image at smaller resolution for speed
            image = cv2.imread(image_path)
            if image is None:
                return None

            # Smaller resize for speed (128x128 instead of 224x224)
            image = cv2.resize(image, (128, 128))

            # Convert to HSV (better for clothing colors)
            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Simple but effective features
            # 1. Color histograms (reduced bins for speed)
            hist_h = cv2.calcHist([image_hsv], [0], None, [16], [0, 180])
            hist_s = cv2.calcHist([image_hsv], [1], None, [16], [0, 256])
            hist_v = cv2.calcHist([image_hsv], [2], None, [16], [0, 256])

            # 2. Simple spatial features (top vs bottom half)
            h, w = image_hsv.shape[:2]
            top_mean = np.mean(image_hsv[: h // 2, :], axis=(0, 1))
            bottom_mean = np.mean(image_hsv[h // 2 :, :], axis=(0, 1))

            # 3. Overall image statistics
            global_mean = np.mean(image_hsv, axis=(0, 1))
            global_std = np.std(image_hsv, axis=(0, 1))

            # Combine features (much smaller feature vector)
            features = np.concatenate(
                [
                    hist_h.flatten(),
                    hist_s.flatten(),
                    hist_v.flatten(),
                    top_mean,
                    bottom_mean,
                    global_mean,
                    global_std,
                ]
            )

            # Normalize
            features = features / (np.linalg.norm(features) + 1e-8)

            return features

        except Exception as e:
            return None

    def _extract_visual_features_basic(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extract visual features from an image for non-face matching.

        Args:
            image_path: Path to the image file

        Returns:
            Feature vector or None if extraction fails
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return None

            # Resize to standard size for consistent feature extraction
            image = cv2.resize(image, (224, 224))

            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Extract color histogram features
            hist_r = cv2.calcHist([image_rgb], [0], None, [64], [0, 256])
            hist_g = cv2.calcHist([image_rgb], [1], None, [64], [0, 256])
            hist_b = cv2.calcHist([image_rgb], [2], None, [64], [0, 256])

            # Extract texture features using LBP (Local Binary Patterns)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Simple texture analysis using gradient magnitude
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(grad_x**2 + grad_y**2)

            # Create feature vector
            color_features = np.concatenate([hist_r.flatten(), hist_g.flatten(), hist_b.flatten()])
            texture_features = np.histogram(magnitude, bins=32)[0]

            # Combine all features
            features = np.concatenate([color_features, texture_features])

            # Normalize features
            features = features / (np.linalg.norm(features) + 1e-8)

            return features

        except Exception as e:
            return None

    def _detect_jersey_number(self, image_path: str) -> Optional[int]:
        """
        Detect jersey number in an image using OCR.

        Args:
            image_path: Path to the image file

        Returns:
            Jersey number if detected, None otherwise
        """
        try:
            # Try to import pytesseract (optional dependency)
            try:
                import pytesseract
            except ImportError:
                return None

            # Load and preprocess image for better OCR
            image = cv2.imread(image_path)
            if image is None:
                return None

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply different preprocessing techniques to improve OCR
            preprocessing_methods = [
                # Original grayscale
                gray,
                # High contrast
                cv2.convertScaleAbs(gray, alpha=2.0, beta=0),
                # Threshold
                cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                # Gaussian blur + threshold
                cv2.threshold(
                    cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )[1],
            ]

            detected_numbers = []

            for processed_image in preprocessing_methods:
                try:
                    # Configure pytesseract for digits only
                    custom_config = r"--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789"
                    text = pytesseract.image_to_string(
                        processed_image, config=custom_config
                    ).strip()

                    # Extract numbers from detected text
                    numbers = re.findall(r"\d+", text)
                    for num_str in numbers:
                        num = int(num_str)
                        # Reasonable jersey number range (1-99)
                        if 1 <= num <= 99:
                            detected_numbers.append(num)

                except Exception:
                    continue

            # Return most common detected number
            if detected_numbers:
                from collections import Counter

                most_common = Counter(detected_numbers).most_common(1)[0][0]
                return most_common

            return None

        except Exception as e:
            return None

    def _load_jersey_number_mapping(self, players_dir: str, logger: Optional[callable] = None):
        """
        Load jersey number mappings from various sources.

        Args:
            players_dir: Directory containing player references
            logger: Optional logging function
        """
        if logger is None:
            logger = print

        # Method 1: Look for jersey_numbers.json file
        json_path = os.path.join(players_dir, "jersey_numbers.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    jersey_mapping = json.load(f)
                    for player_name, jersey_num in jersey_mapping.items():
                        self.player_jersey_numbers[player_name] = int(jersey_num)
                        self.jersey_to_player[int(jersey_num)] = player_name
                logger(f"📋 Loaded jersey numbers from {json_path}")
                return
            except Exception as e:
                logger(f"⚠️  Error loading jersey_numbers.json: {e}")

        # Method 2: Extract from filenames (e.g., "Messi_10.jpg" or "10_Messi.jpg")
        for filename in os.listdir(players_dir):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")):
                continue

            name_without_ext = os.path.splitext(filename)[0]

            # Try pattern: PlayerName_Number
            match = re.match(r"^(.+)_(\d+)$", name_without_ext)
            if match:
                player_name = match.group(1)
                jersey_num = int(match.group(2))
                self.player_jersey_numbers[player_name] = jersey_num
                self.jersey_to_player[jersey_num] = player_name
                continue

            # Try pattern: Number_PlayerName
            match = re.match(r"^(\d+)_(.+)$", name_without_ext)
            if match:
                jersey_num = int(match.group(1))
                player_name = match.group(2)
                self.player_jersey_numbers[player_name] = jersey_num
                self.jersey_to_player[jersey_num] = player_name
                continue

        if self.player_jersey_numbers:
            logger(
                f"🔢 Extracted jersey numbers from filenames: {len(self.player_jersey_numbers)} players"
            )
            logger("⚠️  Note: OCR jersey detection works best with:")
            logger("   - Clear, high-resolution back view photos")
            logger("   - Good lighting and contrast")
            logger("   - Minimal motion blur")
            logger("   - Numbers not obscured by shadows or wrinkles")
        else:
            logger("💡 No jersey numbers found. You can provide them via:")
            logger('   - jersey_numbers.json file: {"Messi": 10, "Ronaldo": 7}')
            logger("   - Filename format: Messi_10.jpg or 10_Messi.jpg")
            logger("📝 Alternative: Use visual feature matching which works better for:")
            logger("   - Action shots, side views, various lighting conditions")

    def load_reference_players(self, players_dir: str, logger: Optional[callable] = None) -> bool:
        """
        Load reference player photos from players/ directory.

        Args:
            players_dir: Path to directory containing player reference photos
            logger: Optional logging function

        Returns:
            True if players loaded successfully
        """
        if logger is None:
            logger = print

        if not os.path.exists(players_dir):
            logger(f"❌ Players directory not found: {players_dir}")
            return False

        if not os.path.isdir(players_dir):
            logger(f"❌ Players path exists but is not a directory: {players_dir}")
            return False

        self.reference_players = {}
        supported_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

        player_files = [
            f for f in os.listdir(players_dir) if f.lower().endswith(supported_extensions)
        ]

        if not player_files:
            logger(f"❌ No player reference photos found in {players_dir}")
            return False

        logger(f"📂 Loading reference players from {players_dir}...")

        # Load jersey number mappings
        if self.enable_jersey_number_matching:
            self._load_jersey_number_mapping(players_dir, logger)

        for player_file in player_files:
            try:
                player_path = os.path.join(players_dir, player_file)
                player_name = os.path.splitext(player_file)[0]

                logger(f"🔍 Processing {player_file} -> player name: '{player_name}'")

                # Load and encode the reference photo
                image = face_recognition.load_image_file(player_path)
                logger(f"  📖 Image loaded: {image.shape if image is not None else 'None'}")

                face_encodings = face_recognition.face_encodings(image)
                logger(f"  🎭 Face encodings found: {len(face_encodings)}")

                if len(face_encodings) == 0:
                    logger(f"⚠️  No face found in {player_file}")
                    continue
                elif len(face_encodings) > 1:
                    logger(f"⚠️  Multiple faces found in {player_file}, using first one")

                self.reference_players[player_name] = face_encodings[0]
                logger(f"✅ Loaded reference for player: {player_name}")

                # Also extract visual features for non-face matching if enabled
                if self.enable_non_face_matching:
                    visual_features = self._extract_visual_features(player_path)
                    if visual_features is not None:
                        self.reference_visual_features[player_name] = visual_features
                        logger(f"  🎨 Visual features extracted for {player_name}")
                    else:
                        logger(f"  ⚠️  Could not extract visual features for {player_name}")

            except Exception as e:
                logger(f"⚠️  Error loading {player_file}: {str(e)}")
                import traceback

                logger(f"  📄 Full error: {traceback.format_exc()}")
                continue

        logger(f"🎯 Loaded {len(self.reference_players)} player face references")
        if self.enable_non_face_matching:
            logger(f"🎨 Loaded {len(self.reference_visual_features)} player visual references")
        return len(self.reference_players) > 0

    def group_photos_by_players(
        self, image_paths: List[str], players_dir: str = None, logger: Optional[callable] = None
    ) -> Dict:
        """
        Group photos by detected players using reference player matching.

        Args:
            image_paths: List of paths to sport photos
            players_dir: Path to directory containing reference player photos
            logger: Optional logging function

        Returns:
            Dictionary with grouping results and statistics
        """
        if logger is None:
            logger = print

        logger("🏃‍♂️ Starting player-based photo grouping...")

        # Load reference players if directory provided
        if players_dir:
            if not self.load_reference_players(players_dir, logger):
                logger("❌ Failed to load reference players")
                return {"success": False, "error": "Failed to load reference players"}
        elif not self.reference_players:
            logger("❌ No reference players loaded and no players directory provided")
            return {"success": False, "error": "No reference players available"}

        # Use parallel processing if enabled and we have enough photos
        if (
            self.use_parallel_processing
            and len(image_paths) >= self.batch_size
            and self.max_worker_threads > 1
        ):

            logger(f"🚀 Processing {len(image_paths)} photos using parallel processing...")
            logger(f"  Threads: {self.max_worker_threads}, Batch size: {self.batch_size}")

            # Process photos in parallel batches
            player_matches = {}  # player_name -> list of photo paths
            multiple_player_photos = []  # photos with multiple recognized players
            unknown_photos = []  # photos with no recognized players

            # Split photos into batches
            batches = []
            for i in range(0, len(image_paths), self.batch_size):
                batch = image_paths[i : i + self.batch_size]
                batches.append((batch, i))  # Include batch index for logging

            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=self.max_worker_threads) as executor:
                # Submit all batches
                future_to_batch = {}
                for batch, batch_idx in batches:
                    future = executor.submit(
                        self._process_photo_batch, batch, batch_idx, len(batches)
                    )
                    future_to_batch[future] = batch_idx

                # Collect results
                for future in as_completed(future_to_batch):
                    batch_idx = future_to_batch[future]
                    try:
                        batch_results = future.result()

                        # Merge batch results
                        for player_name, photos in batch_results["player_matches"].items():
                            if player_name not in player_matches:
                                player_matches[player_name] = []
                            player_matches[player_name].extend(photos)

                        multiple_player_photos.extend(batch_results["multiple_player_photos"])
                        unknown_photos.extend(batch_results["unknown_photos"])

                    except Exception as e:
                        logger(f"❌ Error processing batch {batch_idx}: {e}")

        else:
            # Sequential processing (original method)
            logger(f"🔄 Processing {len(image_paths)} photos sequentially...")

            player_matches = {}  # player_name -> list of photo paths
            multiple_player_photos = []  # photos with multiple recognized players
            unknown_photos = []  # photos with no recognized players

            for i, image_path in enumerate(image_paths):
                try:
                    logger(
                        f"🔍 Analyzing photo {i+1}/{len(image_paths)}: {os.path.basename(image_path)}"
                    )

                    result = self.face_detector.detect_faces(image_path)
                    matched_players = set()

                    if not result.get("face_info") or len(result["face_info"]) == 0:
                        # No faces detected - try alternative matching methods
                        logger(f"  👤 No faces detected, trying alternative matching...")

                        # Try jersey number matching first (most reliable for back views)
                        if self.enable_jersey_number_matching:
                            # Debug: show detected number
                            detected_number = self._detect_jersey_number(image_path)
                            if detected_number:
                                logger(f"  🔢 Detected jersey number: {detected_number}")
                                jersey_match = self.jersey_to_player.get(detected_number)
                                if jersey_match:
                                    matched_players.add(jersey_match)
                                    logger(f"  ✅ Jersey number match found: {jersey_match}")
                                else:
                                    logger(
                                        f"  ❓ Jersey number {detected_number} not found in player mappings"
                                    )
                            else:
                                logger(f"  🔢 No jersey number detected")

                        # Try visual feature matching if no jersey match
                        if not matched_players and self.enable_non_face_matching:
                            visual_match = self._match_visual_features_to_player(image_path)
                            if visual_match:
                                matched_players.add(visual_match)
                                logger(f"  🎨 Visual feature match found: {visual_match}")

                        if not matched_players:
                            logger(f"  ❓ No matches found using any method")
                            unknown_photos.append(image_path)
                            continue
                    else:
                        # Check each face in the photo against reference players
                        for face_info in result["face_info"]:
                            if face_info.get("embedding") is None:
                                continue

                            matched_player = self._match_face_to_player(face_info["embedding"])
                            if matched_player:
                                matched_players.add(matched_player)

                    if len(matched_players) == 0:
                        unknown_photos.append(image_path)
                    elif len(matched_players) == 1:
                        player_name = list(matched_players)[0]
                        if player_name not in player_matches:
                            player_matches[player_name] = []
                        player_matches[player_name].append(image_path)
                    else:
                        # Multiple players detected - assign to all relevant folders
                        multiple_player_photos.append((image_path, list(matched_players)))
                        for player_name in matched_players:
                            if player_name not in player_matches:
                                player_matches[player_name] = []
                            player_matches[player_name].append(image_path)

                except Exception as e:
                    logger(f"⚠️  Error processing {image_path}: {str(e)}")
                    unknown_photos.append(image_path)
                    continue

        # Create output directory structure
        output_dir = os.path.join(self.config.output_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        # Copy photos to player-specific folders
        group_stats = {}
        for player_name, photo_paths in player_matches.items():
            player_dir = os.path.join(output_dir, player_name)
            os.makedirs(player_dir, exist_ok=True)

            copied_count = 0
            for photo_path in photo_paths:
                try:
                    filename = os.path.basename(photo_path)
                    dst_path = os.path.join(player_dir, filename)
                    shutil.copy2(photo_path, dst_path)
                    copied_count += 1
                except Exception as e:
                    logger(f"⚠️  Error copying {photo_path}: {str(e)}")

            group_stats[player_name] = {
                "photo_count": copied_count,
                "folder_path": player_dir,
            }
            logger(f"👤 {player_name}: {copied_count} photos")

        # Handle unknown photos
        if unknown_photos:
            unknown_dir = os.path.join(output_dir, "unknown")
            os.makedirs(unknown_dir, exist_ok=True)

            for photo_path in unknown_photos:
                try:
                    filename = os.path.basename(photo_path)
                    dst_path = os.path.join(unknown_dir, filename)
                    shutil.copy2(photo_path, dst_path)
                except Exception as e:
                    logger(f"⚠️  Error copying unknown photo {photo_path}: {str(e)}")

            logger(f"❓ Unknown: {len(unknown_photos)} photos")

        # Log multiple player detections
        if multiple_player_photos:
            logger(f"👥 Found {len(multiple_player_photos)} photos with multiple players")

        logger(
            f"🎯 Grouping completed! Created {len(player_matches)} player groups + unknown group"
        )

        return {
            "success": True,
            "total_photos": len(image_paths),
            "recognized_players": len(player_matches),
            "photos_with_multiple_players": len(multiple_player_photos),
            "unknown_photos": len(unknown_photos),
            "group_stats": group_stats,
            "output_directory": output_dir,
            "multiple_player_detections": multiple_player_photos,
        }

    def _match_face_to_player(self, face_encoding: np.ndarray) -> Optional[str]:
        """
        Match a face encoding to a reference player.

        Args:
            face_encoding: Face encoding to match

        Returns:
            Player name if match found, None otherwise
        """
        if not self.reference_players or face_encoding is None:
            return None

        best_match = None
        best_similarity = 0.0

        # Compare against ALL reference players to find the BEST match
        for player_name, reference_encoding in self.reference_players.items():
            try:
                # Calculate face distance (lower = more similar)
                distance = face_recognition.face_distance([reference_encoding], face_encoding)[0]

                # Convert distance to similarity (higher = more similar)
                # face_recognition.face_distance returns values where 0.6 is the typical threshold
                similarity = 1.0 - distance

                # Only consider matches above threshold, and track the best one
                if similarity >= self.face_match_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = player_name

                    # Early termination: if we have a very confident match (>0.9 similarity),
                    # we can stop searching as it's unlikely to find a better match
                    if similarity > 0.9:
                        break

            except Exception as e:
                continue

        return best_match

    def _match_visual_features_to_player(self, image_path: str) -> Optional[str]:
        """
        Match a photo to a reference player using visual features (for non-face photos).

        Args:
            image_path: Path to the image to match

        Returns:
            Player name if match found, None otherwise
        """
        if not self.enable_non_face_matching or not self.reference_visual_features:
            return None

        # Extract visual features from the photo
        photo_features = self._extract_visual_features(image_path)
        if photo_features is None:
            return None

        best_match = None
        best_similarity = 0.0

        # Compare against all reference visual features
        for player_name, reference_features in self.reference_visual_features.items():
            try:
                # Calculate cosine similarity
                similarity = cosine_similarity([photo_features], [reference_features])[0][0]

                # Track the best match above threshold
                if similarity >= self.visual_similarity_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = player_name

                    # Early termination: if we have a very confident visual match (>0.95 similarity),
                    # we can stop searching as visual features are less precise than faces
                    if similarity > 0.95:
                        break

            except Exception as e:
                continue

        return best_match

    def _match_jersey_number_to_player(self, image_path: str) -> Optional[str]:
        """
        Match a photo to a reference player using jersey number detection.

        Args:
            image_path: Path to the image to match

        Returns:
            Player name if jersey number match found, None otherwise
        """
        if not self.enable_jersey_number_matching or not self.jersey_to_player:
            return None

        # Detect jersey number in the photo
        jersey_number = self._detect_jersey_number(image_path)
        if jersey_number is None:
            return None

        # Look up player by jersey number
        return self.jersey_to_player.get(jersey_number)

    def _process_photo_batch(
        self, photo_paths: List[str], batch_id: int, total_batches: int
    ) -> Dict:
        """
        Process a batch of photos for player matching with thread safety.

        Args:
            photo_paths: List of photo paths to process
            batch_id: Batch identifier for logging
            total_batches: Total number of batches for progress tracking

        Returns:
            Dictionary with processing results for the batch
        """
        batch_results = {"player_matches": {}, "multiple_player_photos": [], "unknown_photos": []}

        # Process each photo in the batch
        for i, photo_path in enumerate(photo_paths):
            try:
                matched_players = set()

                # Use a more thread-safe approach for face detection
                # Load image manually to avoid potential threading issues
                import face_recognition
                import cv2

                try:
                    # Load image safely
                    image = cv2.imread(photo_path)
                    if image is None:
                        batch_results["unknown_photos"].append(photo_path)
                        continue

                    # Convert BGR to RGB for face_recognition
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Detect faces with thread-safe face_recognition calls
                    face_locations = face_recognition.face_locations(
                        rgb_image, model="hog"
                    )  # hog is more stable in threads

                    if len(face_locations) > 0:
                        # Get face encodings
                        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

                        # Process first 3 faces for speed
                        for face_encoding in face_encodings[:3]:
                            matched_player = self._match_face_to_player_safe(face_encoding)
                            if matched_player:
                                matched_players.add(matched_player)

                    # If no faces found, try alternative matching
                    if not matched_players:
                        if self.enable_jersey_number_matching:
                            jersey_match = self._match_jersey_number_to_player(photo_path)
                            if jersey_match:
                                matched_players.add(jersey_match)

                        if not matched_players and self.enable_non_face_matching:
                            visual_match = self._match_visual_features_to_player_safe(photo_path)
                            if visual_match:
                                matched_players.add(visual_match)

                except Exception:
                    # If any error occurs during processing, treat as unknown
                    batch_results["unknown_photos"].append(photo_path)
                    continue

                # Categorize results
                if len(matched_players) == 0:
                    batch_results["unknown_photos"].append(photo_path)
                elif len(matched_players) == 1:
                    player_name = list(matched_players)[0]
                    if player_name not in batch_results["player_matches"]:
                        batch_results["player_matches"][player_name] = []
                    batch_results["player_matches"][player_name].append(photo_path)
                else:
                    # Multiple players
                    batch_results["multiple_player_photos"].append(
                        (photo_path, list(matched_players))
                    )
                    for player_name in matched_players:
                        if player_name not in batch_results["player_matches"]:
                            batch_results["player_matches"][player_name] = []
                        batch_results["player_matches"][player_name].append(photo_path)

            except Exception:
                batch_results["unknown_photos"].append(photo_path)
                continue

        return batch_results

    def set_face_match_threshold(self, threshold: float):
        """
        Update the face matching threshold.

        Args:
            threshold: New threshold value (0.0 to 1.0, higher = stricter matching)
        """
        self.face_match_threshold = max(0.0, min(1.0, threshold))

    def set_visual_similarity_threshold(self, threshold: float):
        """
        Update the visual similarity threshold for non-face matching.

        Args:
            threshold: New threshold value (0.0 to 1.0, higher = stricter matching)
        """
        self.visual_similarity_threshold = max(0.0, min(1.0, threshold))

    def enable_disable_non_face_matching(self, enabled: bool):
        """
        Enable or disable non-face (visual feature) matching.

        Args:
            enabled: Whether to enable non-face matching
        """
        self.enable_non_face_matching = enabled

    def get_reference_players(self) -> List[str]:
        """Get list of loaded reference player names."""
        return list(self.reference_players.keys())

    def get_visual_reference_players(self) -> List[str]:
        """Get list of players with visual feature references."""
        return list(self.reference_visual_features.keys())
