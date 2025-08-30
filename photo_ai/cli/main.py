"""Main CLI interface for Photo AI."""

import argparse
import os
import sys
from pathlib import Path

from ..core.config import Config
from ..core.photo_processor import PhotoProcessor


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Photo AI - Advanced photo processing and analysis toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  photo-ai process photos/                    # Process all photos in directory
  photo-ai visa input.jpg output.jpg          # Create visa photo
  photo-ai analyze photos/                    # Analyze photo quality
  photo-ai stats                             # Show processing statistics
        """,
    )

    # Global options
    parser.add_argument("--input-dir", "-i", help="Input directory containing photos")
    parser.add_argument(
        "--good-dir", "-g", default="photo-good", help="Directory for good quality photos"
    )
    parser.add_argument(
        "--bad-dir", "-b", default="photo-bad", help="Directory for poor quality photos"
    )
    parser.add_argument(
        "--output-dir", "-o", default="output", help="Output directory for processed photos"
    )
    parser.add_argument("--config", "-c", help="Path to configuration file (not implemented yet)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Process command
    process_parser = subparsers.add_parser("process", help="Process photos with full pipeline")
    process_parser.add_argument(
        "directory", nargs="?", help="Directory containing photos to process"
    )

    # Visa command
    visa_parser = subparsers.add_parser("visa", help="Create visa photo from input image")
    visa_parser.add_argument("input_image", help="Input image file")
    visa_parser.add_argument("output_image", nargs="?", help="Output visa photo file")
    visa_parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode with additional outputs"
    )

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze photo quality without processing"
    )
    analyze_parser.add_argument("directory", help="Directory to analyze")
    analyze_parser.add_argument(
        "--format", choices=["json", "text"], default="text", help="Output format"
    )

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show processing statistics")

    args = parser.parse_args()

    # Create config
    config = Config.from_env()

    # Override config with command line arguments
    if args.input_dir:
        config.input_dir = args.input_dir
    if args.good_dir:
        config.good_dir = args.good_dir
    if args.bad_dir:
        config.bad_dir = args.bad_dir
    if args.output_dir:
        config.output_dir = args.output_dir

    # Create processor
    processor = PhotoProcessor(config)

    try:
        if args.command == "process":
            directory = args.directory or config.input_dir
            if not directory:
                print("Error: No input directory specified")
                return 1

            result = processor.process_photos_pipeline(directory)
            if not result["success"]:
                print(f"Error: {result['error']}")
                return 1

            _print_process_results(result)

        elif args.command == "visa":
            if not os.path.exists(args.input_image):
                print(f"Error: Input image not found: {args.input_image}")
                return 1

            result = processor.process_visa_photo(args.input_image, args.output_image)
            if not result["success"]:
                print(f"Error: {result['error']}")
                return 1

            _print_visa_results(result)

        elif args.command == "analyze":
            if not os.path.exists(args.directory):
                print(f"Error: Directory not found: {args.directory}")
                return 1

            from ..utils.image_utils import get_image_paths

            image_paths = get_image_paths(args.directory)

            if not image_paths:
                print("No images found in directory")
                return 1

            results = processor.analyze_photo_quality(image_paths)
            _print_analysis_results(results, args.format)

        elif args.command == "stats":
            stats = processor.get_processing_stats()
            _print_stats(stats)

        else:
            parser.print_help()
            return 1

    except KeyboardInterrupt:
        print("\\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    return 0


def _print_process_results(result):
    """Print processing results."""
    print("\\n=== Processing Results ===")
    print(f"Input directory: {result['input_dir']}")
    print(f"Total images processed: {result['total_images']}")

    if "sharpness" in result["stages"]:
        stage = result["stages"]["sharpness"]
        print(f"\\nSharpness Analysis:")
        print(f"  Sharp images: {stage['sharp']}")
        print(f"  Blurry images: {stage['blurry']}")

    if "duplicates" in result["stages"] and "skipped" not in result["stages"]["duplicates"]:
        stage = result["stages"]["duplicates"]
        print(f"\\nDuplicate Detection:")
        print(f"  Exact duplicates: {stage['stats']['exact_duplicates_count']}")
        print(f"  Similar images: {stage['stats']['similar_images_count']}")
        print(f"  Unique images (estimate): {stage['stats']['unique_images_estimate']}")

    if "selection" in result["stages"] and "skipped" not in result["stages"]["selection"]:
        stage = result["stages"]["selection"]
        print(f"\\nBest Photo Selection:")
        print(f"  Best photos selected: {stage['best_photos']}")


def _print_visa_results(result):
    """Print visa processing results."""
    print("\\n=== Visa Photo Results ===")
    print(f"Output file: {result['output_path']}")

    dims = result["dimensions"]
    print(
        f"Dimensions: {dims['width_mm']}x{dims['height_mm']}mm ({dims['width_px']}x{dims['height_px']}px)"
    )
    print(f"Resolution: {dims['dpi']} DPI")

    validation = result["validation"]
    if validation["valid"]:
        print("✅ Photo meets visa requirements")
    else:
        print("⚠️ Photo validation issues:")
        for issue in validation["issues"]:
            print(f"  - {issue}")


def _print_analysis_results(results, format_type):
    """Print analysis results."""
    if format_type == "json":
        import json

        print(json.dumps(results, indent=2, default=str))
        return

    print("\\n=== Photo Quality Analysis ===")

    # Sharpness summary
    sharpness_results = results["sharpness"]
    sharp_count = sum(1 for r in sharpness_results.values() if r.get("overall_is_sharp", False))
    print(f"\\nSharpness: {sharp_count}/{len(sharpness_results)} images are sharp")

    # Duplicate summary
    duplicate_stats = results["duplicates"]["stats"]
    print(f"\\nDuplicates:")
    print(f"  Exact duplicates: {duplicate_stats['exact_duplicates_count']} images")
    print(f"  Similar images: {duplicate_stats['similar_images_count']} images")

    # Face analysis summary
    face_results = results["faces"]
    faces_detected = sum(1 for r in face_results.values() if r.get("face_info"))
    print(f"\\nFaces: {faces_detected}/{len(face_results)} images have detectable faces")


def _print_stats(stats):
    """Print processing statistics."""
    print("\\n=== Processing Statistics ===")

    counts = stats["counts"]
    print(f"Input images: {counts['input']}")
    print(f"Good quality: {counts['good']}")
    print(f"Poor quality: {counts['bad']}")
    print(f"Best selected: {counts['best']}")
    print(f"Output files: {counts['output']}")

    print(f"\\nDirectories:")
    for name, path in stats["directories"].items():
        exists = "✅" if os.path.exists(path) else "❌"
        print(f"  {name}: {path} {exists}")


if __name__ == "__main__":
    sys.exit(main())
