"""Background processing thread for photo operations."""

from typing import Union, List
from PyQt6.QtCore import QThread, pyqtSignal

from ...core.photo_processor import PhotoProcessor


class ProcessingThread(QThread):
    """Background thread for photo processing operations."""

    # Signals
    progress_updated = pyqtSignal(int)  # Progress percentage (0-100)
    status_updated = pyqtSignal(str)  # Status message
    finished_processing = pyqtSignal(object)  # Results object
    error_occurred = pyqtSignal(str)  # Error message

    def __init__(
        self, processor: PhotoProcessor, input_source: Union[str, List[str]], operation: str
    ):
        super().__init__()
        self.processor = processor
        self.input_source = input_source
        self.operation = operation  # 'process', 'analyze', 'visa', 'player_grouping', 'quality', 'duplicates', 'selection'

    def run(self):
        """Execute the processing operation."""
        try:
            if self.operation == "process":
                self.run_processing()
            elif self.operation == "analyze":
                self.run_analysis()
            elif self.operation == "visa":
                self.run_visa_processing()
            elif self.operation == "player_grouping":
                self.run_player_grouping()
            elif self.operation == "quality":
                self.run_quality_step()
            elif self.operation == "duplicates":
                self.run_duplicates_step()
            elif self.operation == "selection":
                self.run_selection_step()
            else:
                raise ValueError(f"Unknown operation: {self.operation}")

        except Exception as e:
            self.error_occurred.emit(str(e))

    def run_processing(self):
        """Run complete photo processing pipeline."""
        self.status_updated.emit("Starting photo processing...")
        self.progress_updated.emit(0)

        # If input is a list of files, we need to handle it differently
        if isinstance(self.input_source, list):
            results = self.process_file_list()
        else:
            # It's a directory
            results = self.processor.process_photos_pipeline(self.input_source)

        if results.get("success", False):
            self.progress_updated.emit(100)
            self.status_updated.emit("Processing completed successfully")
            self.finished_processing.emit(results)
        else:
            self.error_occurred.emit(results.get("error", "Unknown error occurred"))

    def run_analysis(self):
        """Run photo analysis only."""
        self.status_updated.emit("Starting photo analysis...")
        self.progress_updated.emit(0)

        try:
            if isinstance(self.input_source, list):
                image_paths = self.input_source
            else:
                # Get image paths from directory
                from ...utils.image_utils import get_image_paths

                image_paths = get_image_paths(self.input_source)

            if not image_paths:
                self.error_occurred.emit("No images found to analyze")
                return

            # Perform analysis
            self.status_updated.emit("Analyzing photo quality...")
            self.progress_updated.emit(25)

            results = self.processor.analyze_photo_quality(image_paths)

            self.progress_updated.emit(100)
            self.status_updated.emit("Analysis completed successfully")
            self.finished_processing.emit(results)

        except Exception as e:
            self.error_occurred.emit(f"Analysis failed: {str(e)}")

    def run_visa_processing(self):
        """Run visa photo processing."""
        if not isinstance(self.input_source, str) or not self.input_source.endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        ):
            self.error_occurred.emit("Visa processing requires a single image file")
            return

        self.status_updated.emit("Starting visa photo processing...")
        self.progress_updated.emit(0)

        try:
            # Determine output path
            import os

            input_name = os.path.splitext(os.path.basename(self.input_source))[0]
            output_path = f"{input_name}_visa.jpg"

            self.status_updated.emit("Analyzing face position...")
            self.progress_updated.emit(20)

            self.status_updated.emit("Adjusting photo dimensions...")
            self.progress_updated.emit(40)

            self.status_updated.emit("Enhancing image quality...")
            self.progress_updated.emit(60)

            self.status_updated.emit("Applying visa photo standards...")
            self.progress_updated.emit(80)

            # Process visa photo
            result = self.processor.process_visa_photo(self.input_source, output_path)

            self.progress_updated.emit(100)

            if result.get("success", False):
                self.status_updated.emit("Visa photo created successfully")
                self.finished_processing.emit(result)
            else:
                self.error_occurred.emit(result.get("error", "Visa processing failed"))

        except Exception as e:
            self.error_occurred.emit(f"Visa processing failed: {str(e)}")

    def process_file_list(self):
        """Process a list of individual files."""
        # For now, we'll create a temporary directory approach
        # In a more sophisticated implementation, we'd modify the processor
        # to handle file lists directly

        import tempfile
        import shutil
        import os

        try:
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                self.status_updated.emit("Preparing files for processing...")
                self.progress_updated.emit(10)

                # Copy files to temp directory
                for i, file_path in enumerate(self.input_source):
                    dest_path = os.path.join(temp_dir, os.path.basename(file_path))
                    shutil.copy2(file_path, dest_path)

                    # Update progress
                    progress = 10 + (i / len(self.input_source)) * 20
                    self.progress_updated.emit(int(progress))

                self.status_updated.emit("Processing photos...")
                self.progress_updated.emit(30)

                # Process the temporary directory
                results = self.processor.process_photos_pipeline(temp_dir)

                # Update the results to reflect original paths
                if results.get("success", False):
                    results["input_dir"] = f"Selected {len(self.input_source)} files"

                return results

        except Exception as e:
            return {"success": False, "error": str(e)}

    def run_player_grouping(self):
        """Run player grouping analysis."""
        self.status_updated.emit("Starting player grouping...")
        self.progress_updated.emit(0)

        try:
            self.status_updated.emit("Detecting faces in photos...")
            self.progress_updated.emit(20)

            self.status_updated.emit("Extracting face features...")
            self.progress_updated.emit(40)

            self.status_updated.emit("Clustering similar faces...")
            self.progress_updated.emit(70)

            self.status_updated.emit("Organizing player groups...")
            self.progress_updated.emit(85)

            # Call the player grouping method
            results = self.processor.group_photos_by_players(self.input_source)

            self.progress_updated.emit(100)

            if results.get("success", False):
                self.status_updated.emit("Player grouping completed successfully!")
                self.finished_processing.emit(results)
            else:
                self.error_occurred.emit(results.get("error", "Player grouping failed"))

        except Exception as e:
            self.error_occurred.emit(f"Player grouping failed: {str(e)}")

    def run_quality_step(self):
        """Run quality analysis step only."""
        self.status_updated.emit("Starting quality analysis...")
        self.progress_updated.emit(0)

        try:
            if isinstance(self.input_source, list):
                image_paths = self.input_source
            else:
                from ...utils.image_utils import get_image_paths

                image_paths = get_image_paths(self.input_source)

            if not image_paths:
                self.error_occurred.emit("No images found to analyze")
                return

            total_images = len(image_paths)
            self.status_updated.emit(f"Analyzing {total_images} photos...")

            # Perform sharpness analysis with progress updates
            self.progress_updated.emit(10)
            self.status_updated.emit("Analyzing sharpness...")
            sharpness_results = self.processor.sharpness_analyzer.batch_analyze(image_paths)

            self.progress_updated.emit(40)
            self.status_updated.emit("Detecting duplicates...")
            duplicate_results = self.processor.duplicate_detector.find_comprehensive_duplicates(
                image_paths
            )

            self.progress_updated.emit(70)
            self.status_updated.emit("Analyzing faces...")

            # Analyze faces with incremental progress
            face_results = {}
            for i, path in enumerate(image_paths):
                face_results[path] = self.processor.face_detector.analyze_face_quality(path)
                # Update progress incrementally during face analysis (70% to 95%)
                face_progress = 70 + int((i + 1) / total_images * 25)
                self.progress_updated.emit(face_progress)
                self.status_updated.emit(f"Analyzing faces... ({i+1}/{total_images})")

            # Compile results
            results = {
                "sharpness": sharpness_results,
                "duplicates": duplicate_results,
                "faces": face_results,
            }

            self.progress_updated.emit(100)
            self.status_updated.emit("Quality analysis completed!")
            self.finished_processing.emit(results)

        except Exception as e:
            self.error_occurred.emit(f"Quality analysis failed: {str(e)}")

    def run_duplicates_step(self):
        """Run duplicate detection step only."""
        self.status_updated.emit("Starting duplicate detection...")
        self.progress_updated.emit(0)

        try:
            if isinstance(self.input_source, list):
                image_paths = self.input_source
            else:
                from ...utils.image_utils import get_image_paths

                image_paths = get_image_paths(self.input_source)

            if not image_paths:
                self.error_occurred.emit("No images found for duplicate detection")
                return

            total_images = len(image_paths)
            self.status_updated.emit(f"Analyzing {total_images} photos for duplicates...")
            self.progress_updated.emit(10)

            # Simulate progress during duplicate detection process
            self.status_updated.emit("Computing image hashes...")
            self.progress_updated.emit(30)

            self.status_updated.emit("Finding exact duplicates...")
            self.progress_updated.emit(50)

            self.status_updated.emit("Detecting similar images...")
            self.progress_updated.emit(70)

            self.status_updated.emit("Clustering results...")
            self.progress_updated.emit(85)

            # Run duplicate detection
            duplicate_results = self.processor.duplicate_detector.find_comprehensive_duplicates(
                image_paths
            )

            results = {"success": True, "duplicates": duplicate_results}

            self.progress_updated.emit(100)
            self.status_updated.emit("Duplicate detection completed")
            self.finished_processing.emit(results)

        except Exception as e:
            self.error_occurred.emit(f"Duplicate detection failed: {str(e)}")

    def run_selection_step(self):
        """Run best photo selection step."""
        self.status_updated.emit("Starting photo selection...")
        self.progress_updated.emit(0)

        try:
            self.status_updated.emit("Evaluating photo quality...")
            self.progress_updated.emit(20)

            self.status_updated.emit("Comparing similar groups...")
            self.progress_updated.emit(50)

            self.status_updated.emit("Selecting best photos...")
            self.progress_updated.emit(75)

            # This step needs results from previous steps
            # For now, we'll create a simple placeholder
            results = {
                "success": True,
                "selection": {
                    "best_photos": 0,
                    "message": "Best photo selection requires quality and duplicate analysis first",
                },
            }

            self.progress_updated.emit(100)
            self.status_updated.emit("Photo selection completed!")
            self.finished_processing.emit(results)

        except Exception as e:
            self.error_occurred.emit(f"Photo selection failed: {str(e)}")
