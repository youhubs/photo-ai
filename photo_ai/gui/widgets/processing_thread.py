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
        self.operation = operation  # 'process', 'analyze', or 'visa'

    def run(self):
        """Execute the processing operation."""
        try:
            if self.operation == "process":
                self.run_processing()
            elif self.operation == "analyze":
                self.run_analysis()
            elif self.operation == "visa":
                self.run_visa_processing()
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

        self.status_updated.emit("Processing visa photo...")
        self.progress_updated.emit(0)

        try:
            # Determine output path
            import os

            input_name = os.path.splitext(os.path.basename(self.input_source))[0]
            output_path = f"{input_name}_visa.jpg"

            self.progress_updated.emit(25)

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
