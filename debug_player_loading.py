#!/usr/bin/env python3
"""Debug script to test player reference loading."""

import sys
import os

sys.path.insert(0, "/Users/tigerhu/projects/photo-ai")

from photo_ai.core.config import Config
from photo_ai.processors.sports.player_grouping import PlayerGroupingProcessor


def debug_player_loading():
    """Debug the player reference loading process."""
    print("🔍 Debugging Player Reference Loading")
    print("=" * 50)

    # Initialize the system
    config = Config()
    processor = PlayerGroupingProcessor(config)

    players_dir = "players"
    print(f"📂 Checking players directory: {players_dir}")

    if not os.path.exists(players_dir):
        print(f"❌ Directory doesn't exist: {players_dir}")
        return

    # List all files in players directory
    files = os.listdir(players_dir)
    image_files = [
        f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"))
    ]

    print(f"📸 Found {len(image_files)} image files:")
    for f in sorted(image_files):
        print(f"  - {f}")

    print("\n🔄 Loading reference players...")

    # Try to load reference players with detailed logging
    success = processor.load_reference_players(players_dir)

    print(f"\n📊 Loading Summary:")
    print(f"  Success: {success}")
    print(f"  Players loaded: {len(processor.reference_players)}")

    if processor.reference_players:
        print(f"\n✅ Successfully loaded players:")
        for name in sorted(processor.reference_players.keys()):
            print(f"  - {name}")
    else:
        print("❌ No players were successfully loaded!")

    print(f"\n🎯 Current face match threshold: {processor.face_match_threshold}")

    return processor


if __name__ == "__main__":
    processor = debug_player_loading()
