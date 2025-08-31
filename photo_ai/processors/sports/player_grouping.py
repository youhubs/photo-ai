"""Player-based photo grouping for sports photos."""

import os
import shutil
from typing import Dict, List, Optional
import numpy as np

from ...core.config import Config
from ..face.detector import FaceDetector


class PlayerGroupingProcessor:
    """Groups sports photos by detected players using face recognition."""

    def __init__(self, config: Config):
        self.config = config
        self.face_detector = FaceDetector(config)

    def group_photos_by_players(
        self, image_paths: List[str], logger: Optional[callable] = None
    ) -> Dict:
        """
        Group photos by detected players using face clustering.

        Args:
            image_paths: List of paths to sport photos
            logger: Optional logging function

        Returns:
            Dictionary with grouping results and statistics
        """
        if logger is None:
            logger = print

        logger("🏃‍♂️ Starting player-based photo grouping...")

        # Extract face embeddings from all photos
        face_data = []
        valid_photos = []

        for i, image_path in enumerate(image_paths):
            try:
                logger(
                    f"🔍 Analyzing faces in photo {i+1}/{len(image_paths)}: {os.path.basename(image_path)}"
                )

                result = self.face_detector.detect_faces(image_path)
                if result.get("face_info") and len(result["face_info"]) > 0:
                    # For sports photos, we typically want the main/largest face
                    main_face = max(result["face_info"], key=lambda x: x.get("confidence", 0))

                    if main_face.get("embedding") is not None:
                        face_data.append(
                            {
                                "image_path": image_path,
                                "embedding": main_face["embedding"],
                                "confidence": main_face.get("confidence", 0),
                                "face_count": len(result["face_info"]),
                            }
                        )
                        valid_photos.append(image_path)

            except Exception as e:
                logger(f"⚠️  Error processing {image_path}: {str(e)}")
                continue

        if len(face_data) < 2:
            logger("❌ Not enough faces detected for meaningful grouping")
            return {
                "success": False,
                "error": "Not enough faces detected for grouping",
                "photos_with_faces": len(face_data),
                "total_photos": len(image_paths),
            }

        logger(f"✅ Found faces in {len(face_data)}/{len(image_paths)} photos")

        # Cluster faces by similarity
        player_groups = self._cluster_faces(face_data, logger)

        # Create output directory structure
        groups_output_dir = os.path.join(self.config.output_dir, "player_groups")
        os.makedirs(groups_output_dir, exist_ok=True)

        # Copy photos to player-specific folders
        group_stats = {}
        for group_id, group_photos in player_groups.items():
            group_dir = os.path.join(groups_output_dir, f"player_{group_id:02d}")
            os.makedirs(group_dir, exist_ok=True)

            copied_count = 0
            for photo_info in group_photos:
                try:
                    src_path = photo_info["image_path"]
                    filename = os.path.basename(src_path)
                    dst_path = os.path.join(group_dir, filename)
                    shutil.copy2(src_path, dst_path)
                    copied_count += 1
                except Exception as e:
                    logger(f"⚠️  Error copying {src_path}: {str(e)}")

            group_stats[group_id] = {
                "photo_count": copied_count,
                "avg_confidence": np.mean([p["confidence"] for p in group_photos]),
                "folder_path": group_dir,
            }

            logger(
                f"👤 Player {group_id}: {copied_count} photos (avg confidence: {group_stats[group_id]['avg_confidence']:.2f})"
            )

        # Handle ungrouped photos (no clear face detected)
        ungrouped_photos = [p for p in image_paths if p not in valid_photos]
        if ungrouped_photos:
            ungrouped_dir = os.path.join(groups_output_dir, "ungrouped")
            os.makedirs(ungrouped_dir, exist_ok=True)

            for photo_path in ungrouped_photos:
                try:
                    filename = os.path.basename(photo_path)
                    dst_path = os.path.join(ungrouped_dir, filename)
                    shutil.copy2(photo_path, dst_path)
                except Exception as e:
                    logger(f"⚠️  Error copying ungrouped photo {photo_path}: {str(e)}")

        logger(f"🎯 Grouping completed! Created {len(player_groups)} player groups")

        return {
            "success": True,
            "total_photos": len(image_paths),
            "photos_with_faces": len(face_data),
            "player_groups": len(player_groups),
            "ungrouped_photos": len(ungrouped_photos),
            "group_stats": group_stats,
            "output_directory": groups_output_dir,
        }

    def _cluster_faces(self, face_data: List[Dict], logger: callable) -> Dict[int, List[Dict]]:
        """
        Cluster face embeddings using similarity threshold.

        Args:
            face_data: List of face data with embeddings
            logger: Logging function

        Returns:
            Dictionary mapping group_id to list of photo info
        """
        logger("🔄 Clustering faces by similarity...")

        if not face_data:
            return {}

        # Simple clustering based on cosine similarity
        embeddings = np.array([item["embedding"] for item in face_data])

        # Normalize embeddings
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Compute pairwise similarities
        similarities = np.dot(embeddings, embeddings.T)

        # Threshold for considering faces as same person (adjust as needed)
        similarity_threshold = 0.6

        # Simple clustering algorithm
        groups = {}
        assigned = set()
        group_id = 0

        for i, face_info in enumerate(face_data):
            if i in assigned:
                continue

            # Start new group
            current_group = [face_info]
            assigned.add(i)

            # Find similar faces
            for j, other_face_info in enumerate(face_data):
                if j in assigned or j == i:
                    continue

                if similarities[i, j] >= similarity_threshold:
                    current_group.append(other_face_info)
                    assigned.add(j)

            groups[group_id] = current_group
            group_id += 1

        # Merge small groups with confidence < 0.7 into larger ones if similar enough
        merged_groups = self._merge_small_groups(groups, similarities, face_data, logger)

        return merged_groups

    def _merge_small_groups(
        self, groups: Dict, similarities: np.ndarray, face_data: List[Dict], logger: callable
    ) -> Dict[int, List[Dict]]:
        """Merge small groups with larger ones if they're similar enough."""

        # Find groups with < 3 photos and low average confidence
        small_groups = []
        large_groups = []

        for group_id, group_photos in groups.items():
            avg_confidence = np.mean([p["confidence"] for p in group_photos])

            if len(group_photos) < 3 and avg_confidence < 0.7:
                small_groups.append(group_id)
            else:
                large_groups.append(group_id)

        # Try to merge small groups with large ones
        merged_groups = {gid: groups[gid] for gid in large_groups}
        next_group_id = max(large_groups) + 1 if large_groups else 0

        for small_group_id in small_groups:
            small_group = groups[small_group_id]
            merged = False

            # Try to merge with existing large groups
            for large_group_id in large_groups:
                # Check similarity between group representatives
                small_embedding = small_group[0]["embedding"]
                large_embedding = merged_groups[large_group_id][0]["embedding"]

                # Normalize and compute similarity
                small_norm = small_embedding / np.linalg.norm(small_embedding)
                large_norm = large_embedding / np.linalg.norm(large_embedding)
                similarity = np.dot(small_norm, large_norm)

                if similarity >= 0.5:  # Lower threshold for merging
                    merged_groups[large_group_id].extend(small_group)
                    merged = True
                    break

            # If not merged, keep as separate group
            if not merged:
                merged_groups[next_group_id] = small_group
                next_group_id += 1

        return merged_groups
