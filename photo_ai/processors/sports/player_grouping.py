"""Player-based photo grouping for sports photos with reference player matching."""

import os
import shutil
from typing import Dict, List, Optional, Tuple
import numpy as np
import face_recognition
from pathlib import Path

from ...core.config import Config
from ..face.detector import FaceDetector


class PlayerGroupingProcessor:
    """Groups sports photos by detected players using reference player photos."""

    def __init__(self, config: Config):
        self.config = config
        self.face_detector = FaceDetector(config)
        self.reference_players = {}  # Will store player_name -> encoding mapping
        self.face_match_threshold = config.processing.face_match_threshold

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

        for player_file in player_files:
            try:
                player_path = os.path.join(players_dir, player_file)
                player_name = os.path.splitext(player_file)[0]

                # Load and encode the reference photo
                image = face_recognition.load_image_file(player_path)
                face_encodings = face_recognition.face_encodings(image)

                if len(face_encodings) == 0:
                    logger(f"⚠️  No face found in {player_file}")
                    continue
                elif len(face_encodings) > 1:
                    logger(f"⚠️  Multiple faces found in {player_file}, using first one")

                self.reference_players[player_name] = face_encodings[0]
                logger(f"✅ Loaded reference for player: {player_name}")

            except Exception as e:
                logger(f"⚠️  Error loading {player_file}: {str(e)}")
                continue

        logger(f"🎯 Loaded {len(self.reference_players)} player references")
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

        # Process each photo and match against reference players
        player_matches = {}  # player_name -> list of photo paths
        multiple_player_photos = []  # photos with multiple recognized players
        unknown_photos = []  # photos with no recognized players

        for i, image_path in enumerate(image_paths):
            try:
                logger(
                    f"🔍 Analyzing photo {i+1}/{len(image_paths)}: {os.path.basename(image_path)}"
                )

                result = self.face_detector.detect_faces(image_path)
                if not result.get("face_info") or len(result["face_info"]) == 0:
                    unknown_photos.append(image_path)
                    continue

                # Check each face in the photo against reference players
                matched_players = set()
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

        # Compare against all reference players
        for player_name, reference_encoding in self.reference_players.items():
            try:
                # Calculate face distance (lower = more similar)
                distance = face_recognition.face_distance([reference_encoding], face_encoding)[0]

                # Convert distance to similarity (higher = more similar)
                # face_recognition.face_distance returns values where 0.6 is the typical threshold
                similarity = 1.0 - distance

                if similarity >= self.face_match_threshold:
                    return player_name

            except Exception as e:
                continue

        return None

    def set_face_match_threshold(self, threshold: float):
        """
        Update the face matching threshold.

        Args:
            threshold: New threshold value (0.0 to 1.0, higher = stricter matching)
        """
        self.face_match_threshold = max(0.0, min(1.0, threshold))

    def get_reference_players(self) -> List[str]:
        """Get list of loaded reference player names."""
        return list(self.reference_players.keys())
