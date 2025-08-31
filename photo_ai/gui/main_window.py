"""Main window for Photo AI desktop application."""

import os
from typing import Optional, List
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QPushButton,
    QLabel,
    QTextEdit,
    QProgressBar,
    QTabWidget,
    QListWidget,
    QFileDialog,
    QMessageBox,
    QSplitter,
    QButtonGroup,
    QRadioButton,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QKeySequence, QShortcut

from ..core.photo_processor import PhotoProcessor
from ..core.config import Config
from .dialogs.visa_dialog import VisaPhotoDialog
from .dialogs.settings_dialog import SettingsDialog
from .widgets.photo_viewer import PhotoViewer
from .widgets.processing_thread import ProcessingThread
from .widgets.logger_widget import LoggerWidget


class ImageCounterThread(QThread):
    """Background thread for counting images in a folder."""

    count_finished = pyqtSignal(int, list)  # count, first_few_paths

    def __init__(self, folder_path: str):
        super().__init__()
        self.folder_path = folder_path

    def run(self):
        """Count images and get first few paths for preview."""
        try:
            from ..utils.image_utils import get_image_paths

            image_paths = get_image_paths(self.folder_path)

            # Get first 10 images for quick preview
            preview_paths = image_paths[:10] if len(image_paths) > 10 else image_paths

            self.count_finished.emit(len(image_paths), preview_paths)
        except Exception as e:
            # Emit 0 count on error
            self.count_finished.emit(0, [])


class PhotoAIMainWindow(QMainWindow):
    """Main window for the Photo AI desktop application."""

    def __init__(self):
        super().__init__()

        # Initialize processor
        self.config = Config.from_env()
        self.processor = PhotoProcessor(self.config)

        # Processing state
        self.current_processing_thread: Optional[ProcessingThread] = None
        self.selected_folder: Optional[str] = None
        self.selected_files: List[str] = []
        self.current_mode = "batch"  # "batch" or "single"

        # Step-by-step processing state
        self.step_results = {}  # Store results from each step
        self.completed_steps = set()  # Track which steps are done

        # Async loading state
        self.image_counter_thread: Optional[ImageCounterThread] = None

        # Setup UI
        self.setup_ui()
        self.setup_connections()

    def setup_ui(self):
        """Setup the main user interface."""
        self.setWindowTitle("Photo AI - Intelligent Photo Processing")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel - Controls
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)

        # Right panel - Photo viewer and results
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)

        # Set splitter proportions
        splitter.setSizes([400, 800])

        # Status bar
        self.statusBar().showMessage("Ready to process photos")

        # Menu bar
        self.create_menu_bar()

    def create_left_panel(self) -> QWidget:
        """Create the left control panel."""
        panel = QWidget()
        panel.setMaximumWidth(450)
        layout = QVBoxLayout(panel)

        # App title
        title = QLabel("Photo AI")
        title.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("Intelligent Photo Processing")
        subtitle.setFont(QFont("Arial", 12))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #888888; margin-bottom: 10px;")
        layout.addWidget(subtitle)

        # Mode selection
        mode_group = QGroupBox("Processing Mode")
        mode_layout = QHBoxLayout(mode_group)

        self.mode_button_group = QButtonGroup()

        self.batch_mode_radio = QRadioButton("🏃‍♂️ Batch Mode")
        self.batch_mode_radio.setToolTip("Sports photo processing - batch curation workflow")
        self.batch_mode_radio.setChecked(True)
        mode_layout.addWidget(self.batch_mode_radio)
        self.mode_button_group.addButton(self.batch_mode_radio, 0)

        self.single_mode_radio = QRadioButton("📄 Single Mode")
        self.single_mode_radio.setToolTip("Visa photo processing - single photo workflow")
        mode_layout.addWidget(self.single_mode_radio)
        self.mode_button_group.addButton(self.single_mode_radio, 1)

        layout.addWidget(mode_group)

        # Photo selection group
        self.selection_group = QGroupBox("Photo Selection")
        selection_layout = QVBoxLayout(self.selection_group)

        self.select_folder_btn = QPushButton("📁 Select Photo Folder")
        self.select_folder_btn.setMinimumHeight(50)
        self.select_folder_btn.setToolTip("Select a folder containing photos to process")
        selection_layout.addWidget(self.select_folder_btn)

        self.select_files_btn = QPushButton("🖼️ Select Individual Photos")
        self.select_files_btn.setMinimumHeight(50)
        self.select_files_btn.setToolTip("Select specific photo files to process")
        selection_layout.addWidget(self.select_files_btn)

        self.select_single_btn = QPushButton("🖼️ Select Portrait Photo")
        self.select_single_btn.setMinimumHeight(50)
        self.select_single_btn.setToolTip("Select a single portrait photo for visa processing")
        self.select_single_btn.setVisible(False)  # Hidden in batch mode
        selection_layout.addWidget(self.select_single_btn)

        self.selected_path_label = QLabel("No folder or files selected")
        self.selected_path_label.setWordWrap(True)
        self.selected_path_label.setStyleSheet(
            "color: #666666; padding: 10px; border: 1px solid #444444; border-radius: 5px;"
        )
        selection_layout.addWidget(self.selected_path_label)

        layout.addWidget(self.selection_group)

        # Processing options
        self.processing_group = QGroupBox("Processing Options")
        processing_layout = QVBoxLayout(self.processing_group)

        # Step-by-step processing buttons (always visible)
        self.step1_btn = QPushButton("1️⃣ Quality Analysis")
        self.step1_btn.setMinimumHeight(40)
        self.step1_btn.setEnabled(False)
        self.step1_btn.setToolTip("Analyze photo sharpness and quality")
        processing_layout.addWidget(self.step1_btn)

        self.step2_btn = QPushButton("2️⃣ Duplicate Detection")
        self.step2_btn.setMinimumHeight(40)
        self.step2_btn.setEnabled(False)
        self.step2_btn.setToolTip("Find and group duplicate/similar photos")
        processing_layout.addWidget(self.step2_btn)

        self.step3_btn = QPushButton("3️⃣ Best Photo Selection")
        self.step3_btn.setMinimumHeight(40)
        self.step3_btn.setEnabled(False)
        self.step3_btn.setToolTip("Select best photos from each group")
        processing_layout.addWidget(self.step3_btn)

        self.step4_btn = QPushButton("4️⃣ Player Grouping")
        self.step4_btn.setMinimumHeight(40)
        self.step4_btn.setEnabled(False)
        self.step4_btn.setToolTip("Group photos by detected players")
        processing_layout.addWidget(self.step4_btn)

        # Cancel button (initially hidden)
        self.cancel_btn = QPushButton("❌ Cancel Processing")
        self.cancel_btn.setMinimumHeight(35)
        self.cancel_btn.setVisible(False)
        self.cancel_btn.setToolTip("Cancel the current processing step")
        self.cancel_btn.setStyleSheet(
            "QPushButton { background-color: #ff4444; color: white; font-weight: bold; }"
        )
        processing_layout.addWidget(self.cancel_btn)

        # Single mode buttons (initially hidden)
        self.visa_btn = QPushButton("📄 Create Visa Photo")
        self.visa_btn.setMinimumHeight(50)
        self.visa_btn.setToolTip("Create a visa/passport photo from selected image")
        self.visa_btn.setVisible(False)
        processing_layout.addWidget(self.visa_btn)

        self.enhance_btn = QPushButton("✨ Enhance Portrait")
        self.enhance_btn.setMinimumHeight(40)
        self.enhance_btn.setToolTip("Enhance the selected portrait photo")
        self.enhance_btn.setVisible(False)
        processing_layout.addWidget(self.enhance_btn)

        layout.addWidget(self.processing_group)

        # Progress section
        progress_group = QGroupBox("Processing Progress")
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("")
        self.progress_label.setVisible(False)
        progress_layout.addWidget(self.progress_label)

        layout.addWidget(progress_group)

        # Results summary
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(200)
        self.results_text.setPlaceholderText("Processing results will appear here...")
        layout.addWidget(self.results_text)

        # Settings button
        self.settings_btn = QPushButton("⚙️ Settings")
        layout.addWidget(self.settings_btn)

        layout.addStretch()
        return panel

    def create_right_panel(self) -> QWidget:
        """Create the right panel with tabs."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # Photo viewer
        self.photo_viewer = PhotoViewer()
        self.tab_widget.addTab(self.photo_viewer, "📷 Photo Viewer")

        # Results tab
        self.results_widget = QTextEdit()
        self.results_widget.setReadOnly(True)
        self.tab_widget.addTab(self.results_widget, "📊 Results")

        # Logger tab with dark-mode
        self.logger = LoggerWidget(dark_mode=True)
        self.tab_widget.addTab(self.logger, "📝 Log")

        return panel

    def create_menu_bar(self):
        """Create menu bar."""
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        file_menu.addAction("Select Folder", self.select_folder)
        file_menu.addAction("Select Files", self.select_files)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        tools_menu = menubar.addMenu("Tools")
        tools_menu.addAction("Create Visa Photo", self.open_visa_dialog)
        tools_menu.addAction("Settings", self.open_settings)

        help_menu = menubar.addMenu("Help")
        help_menu.addAction("About", self.show_about)

    def setup_connections(self):
        # Mode switching
        self.mode_button_group.idClicked.connect(self.on_mode_changed)

        # Photo selection
        self.select_folder_btn.clicked.connect(self.select_folder)
        self.select_files_btn.clicked.connect(self.select_files)
        self.select_single_btn.clicked.connect(self.select_single_photo)

        # Processing actions - direct step execution
        self.step1_btn.clicked.connect(lambda: self.execute_step("quality"))
        self.step2_btn.clicked.connect(lambda: self.execute_step("duplicates"))
        self.step3_btn.clicked.connect(lambda: self.execute_step("selection"))
        self.step4_btn.clicked.connect(lambda: self.execute_step("player_grouping"))
        self.cancel_btn.clicked.connect(self.cancel_processing)
        self.visa_btn.clicked.connect(self.open_visa_dialog)
        self.enhance_btn.clicked.connect(self.enhance_portrait)

        # Settings
        self.settings_btn.clicked.connect(self.open_settings)

        # Add keyboard shortcut for canceling (Escape key)
        self.cancel_shortcut = QShortcut(QKeySequence.StandardKey.Cancel, self)
        self.cancel_shortcut.activated.connect(self.cancel_processing)

    def on_mode_changed(self, mode_id: int):
        """Handle mode switching between batch and single processing."""
        if mode_id == 0:  # Batch mode
            self.current_mode = "batch"
            self.update_ui_for_batch_mode()
        else:  # Single mode
            self.current_mode = "single"
            self.update_ui_for_single_mode()

        # Clear current selections when switching modes
        self.selected_folder = None
        self.selected_files = []
        self.update_selection_display()

    def update_ui_for_batch_mode(self):
        """Update UI elements for batch processing mode."""
        # Update group titles
        self.selection_group.setTitle("Sports Photo Selection")
        self.processing_group.setTitle("Processing Steps")

        # Show/hide appropriate buttons
        self.select_folder_btn.setVisible(True)
        self.select_files_btn.setVisible(True)
        self.select_single_btn.setVisible(False)

        # Show step buttons directly
        self.step1_btn.setVisible(True)
        self.step2_btn.setVisible(True)
        self.step3_btn.setVisible(True)
        self.step4_btn.setVisible(True)
        self.cancel_btn.setVisible(False)  # Hidden unless processing

        # Hide single mode buttons
        self.visa_btn.setVisible(False)
        self.enhance_btn.setVisible(False)

        self.selected_path_label.setText("No sports photos selected")

        # Initialize results text for step-by-step processing
        if not hasattr(self, "results_text_initialized"):
            self.results_text.setText(
                "📋 Processing Steps Ready\n\n"
                "Select photos and click any step to begin:\n"
                "1️⃣ Quality Analysis\n"
                "2️⃣ Duplicate Detection\n"
                "3️⃣ Best Photo Selection (needs 1 & 2)\n"
                "4️⃣ Player Grouping"
            )
            self.results_text_initialized = True

    def update_ui_for_single_mode(self):
        """Update UI elements for single photo processing mode."""
        # Update group titles
        self.selection_group.setTitle("Portrait Photo Selection")
        self.processing_group.setTitle("Single Photo Processing")

        # Show/hide appropriate buttons
        self.select_folder_btn.setVisible(False)
        self.select_files_btn.setVisible(False)
        self.select_single_btn.setVisible(True)

        # Hide batch mode step buttons
        self.step1_btn.setVisible(False)
        self.step2_btn.setVisible(False)
        self.step3_btn.setVisible(False)
        self.step4_btn.setVisible(False)
        self.cancel_btn.setVisible(False)

        # Show single mode buttons
        self.visa_btn.setVisible(True)
        self.enhance_btn.setVisible(True)

        self.selected_path_label.setText("No portrait photo selected")

    def select_single_photo(self):
        """Select a single photo for visa/portrait processing."""
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Select Portrait Photo",
            "",
            "Image Files (*.jpg *.jpeg *.png *.bmp *.tiff *.webp);;All Files (*)",
        )
        if file:
            self.selected_files = [file]
            self.selected_folder = None
            self.update_selection_display()
            self.load_photos_preview_async()

    def execute_step(self, step_name: str):
        """Execute a specific processing step."""
        if self.current_processing_thread and self.current_processing_thread.isRunning():
            return

        input_source = self.selected_folder if self.selected_folder else self.selected_files

        # Create processing thread for this specific step
        self.current_processing_thread = ProcessingThread(self.processor, input_source, step_name)
        self.current_processing_thread.progress_updated.connect(self.update_progress)
        self.current_processing_thread.status_updated.connect(self.update_status)
        self.current_processing_thread.finished_processing.connect(
            lambda results: self.on_step_completed(step_name, results)
        )
        self.current_processing_thread.error_occurred.connect(self.on_processing_error)

        # Update UI for running step
        self.show_progress(True)
        self.disable_step_buttons()
        self.show_cancel_button(True)
        self.update_step_button_status(step_name, "running")

        self.current_processing_thread.start()
        self.log_message(f"Started {step_name} step...", "info")

    def disable_step_buttons(self):
        """Disable all step buttons during processing."""
        self.step1_btn.setEnabled(False)
        self.step2_btn.setEnabled(False)
        self.step3_btn.setEnabled(False)
        self.step4_btn.setEnabled(False)

    def show_cancel_button(self, show: bool):
        """Show or hide the cancel button."""
        self.cancel_btn.setVisible(show)

    def cancel_processing(self):
        """Cancel the current processing operation."""
        if self.current_processing_thread and self.current_processing_thread.isRunning():
            # Ask user to confirm cancellation for long-running processes
            reply = QMessageBox.question(
                self,
                "Cancel Processing",
                "Are you sure you want to cancel the current processing step?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.current_processing_thread.terminate()
                self.current_processing_thread.wait()  # Wait for thread to finish

                # Reset UI
                self.show_progress(False)
                self.show_cancel_button(False)
                self.enable_available_steps()

                # Reset any running button status
                self.reset_step_button_statuses()

                self.results_text.setText(
                    "❌ Processing cancelled by user\n\nYou can restart any step when ready."
                )
                self.statusBar().showMessage("Processing cancelled")
                self.log_message("Processing cancelled by user", "warning")
        else:
            # No processing running, just hide cancel button if visible
            self.show_cancel_button(False)

    def reset_step_button_statuses(self):
        """Reset step button text to original state."""
        self.step1_btn.setText("1️⃣ Quality Analysis")
        self.step2_btn.setText("2️⃣ Duplicate Detection")
        self.step3_btn.setText("3️⃣ Best Photo Selection")
        self.step4_btn.setText("4️⃣ Player Grouping")

    def update_step_button_status(self, step_name: str, status: str):
        """Update the visual status of step buttons."""
        step_buttons = {
            "quality": (self.step1_btn, "Quality Analysis"),
            "duplicates": (self.step2_btn, "Duplicate Detection"),
            "selection": (self.step3_btn, "Best Photo Selection"),
            "player_grouping": (self.step4_btn, "Player Grouping"),
        }

        if step_name in step_buttons:
            btn, name = step_buttons[step_name]
            if status == "running":
                btn.setText(f"{btn.text().split(' ')[0]} {name} ⏳")
            elif status == "completed":
                btn.setText(f"{btn.text().split(' ')[0]} {name} ✅")
            elif status == "ready":
                btn.setText(f"{btn.text().split(' ')[0]} {name} ▶️")

    def on_step_completed(self, step_name: str, results):
        """Handle completion of a processing step."""
        self.show_progress(False)
        self.show_cancel_button(False)

        # Store step results
        self.step_results[step_name] = results
        self.completed_steps.add(step_name)

        # Update button status
        self.update_step_button_status(step_name, "completed")

        # Enable next step(s) and current step for re-running
        self.enable_available_steps()

        # Display step results
        if results.get("success", False):
            self.display_step_results(step_name, results)
            self.statusBar().showMessage(
                f"{step_name.replace('_', ' ').title()} completed successfully"
            )
            self.log_message(
                f"{step_name.replace('_', ' ').title()} completed successfully", "success"
            )
        else:
            error_msg = results.get("error", f"{step_name} failed")
            self.statusBar().showMessage(error_msg)
            self.log_message(f"Error in {step_name}: {error_msg}", "error")

    def enable_available_steps(self):
        """Enable steps that can be run based on completed steps."""
        has_photos = self.selected_folder or self.selected_files

        if not has_photos:
            return

        # Steps 1, 2, and 4 can run independently when photos are available
        self.step1_btn.setEnabled(True)
        self.step2_btn.setEnabled(True)
        self.step4_btn.setEnabled(True)

        # Step 3 requires steps 1 & 2 to be completed
        self.update_step3_availability()

        # Update ready status for step 3 if it just became available
        if self.step3_btn.isEnabled() and "selection" not in self.completed_steps:
            self.update_step_button_status("selection", "ready")

    def display_step_results(self, step_name: str, results):
        """Display results for a specific step."""
        # Map step names to display methods
        if step_name == "quality":
            self.display_quality_step_results(results)
        elif step_name == "duplicates":
            self.display_duplicates_step_results(results)
        elif step_name == "selection":
            self.display_selection_step_results(results)
        elif step_name == "player_grouping":
            self.display_player_grouping_results(results)

        # Switch to Results tab to show the new results
        self.tab_widget.setCurrentIndex(1)

    def display_quality_step_results(self, results):
        """Display quality analysis step results."""
        if not results.get("success", False):
            self.display_step_error("Quality Analysis", results.get("error", "Unknown error"))
            return

        # Simple summary for left panel
        sharp_count = sum(
            1 for r in results.get("sharpness", {}).values() if r.get("overall_is_sharp", False)
        )
        total_count = len(results.get("sharpness", {}))

        progress_text = f"1️⃣ Quality Analysis completed!\n\n📷 {total_count} photos analyzed"
        progress_text += (
            f"\n✨ {sharp_count} sharp photos ({(sharp_count/total_count*100):.0f}%)"
            if total_count > 0
            else ""
        )
        progress_text += f"\n🌫️ {total_count - sharp_count} blurry photos"
        progress_text += "\n\n📊 See Results tab for details"

        self.results_text.setText(progress_text)

        # Detailed results for Results tab
        self.display_analysis_results(results)

    def display_duplicates_step_results(self, results):
        """Display duplicate detection step results."""
        if not results.get("success", False):
            self.display_step_error("Duplicate Detection", results.get("error", "Unknown error"))
            return

        stats = results.get("duplicates", {}).get("stats", {})
        exact_dups = stats.get("exact_duplicates_count", 0)
        similar_imgs = stats.get("similar_images_count", 0)
        unique_imgs = stats.get("unique_images_estimate", 0)

        progress_text = (
            f"2️⃣ Duplicate Detection completed!\n\n🔗 {exact_dups} exact duplicates found"
        )
        progress_text += f"\n🎭 {similar_imgs} similar images found"
        progress_text += f"\n✨ {unique_imgs} unique images estimated"
        progress_text += "\n\n📊 See Results tab for details"

        self.results_text.setText(progress_text)

        # Display detailed results (reuse existing analysis display)
        self.display_analysis_results(results)

    def display_selection_step_results(self, results):
        """Display best photo selection step results."""
        if not results.get("success", False):
            self.display_step_error("Best Photo Selection", results.get("error", "Unknown error"))
            return

        best_photos = results.get("selection", {}).get("best_photos", 0)

        progress_text = (
            f"3️⃣ Best Photo Selection completed!\n\n🏆 {best_photos} best photos selected"
        )
        progress_text += "\n📁 Photos copied to output directory"
        progress_text += "\n\n📊 See Results tab for details"

        self.results_text.setText(progress_text)

        # Display detailed results
        self.display_analysis_results(results)

    def display_step_error(self, step_name: str, error: str):
        """Display error for a failed step."""
        self.results_text.setText(
            f"❌ {step_name} failed!\n\nError: {error}\n\n📝 Check Log tab for details"
        )

        error_html = f"""
        <div style='color: #ff6b6b; background: #2a1a1a; padding: 15px; border-radius: 8px; border-left: 4px solid #ff6b6b;'>
            <h3 style='margin: 0 0 10px 0;'>❌ {step_name} Failed</h3>
            <p style='margin: 0;'>{error}</p>
        </div>
        """
        self.results_widget.setHtml(error_html)

    def enhance_portrait(self):
        """Enhance the selected portrait photo."""
        if not self.selected_files:
            QMessageBox.warning(self, "No Photo Selected", "Please select a portrait photo first.")
            return

        QMessageBox.information(
            self,
            "Portrait Enhancement",
            "Portrait enhancement will improve lighting, contrast, and overall quality.\n\n"
            "This feature is coming soon!",
        )

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Photo Folder", "", QFileDialog.Option.ShowDirsOnly
        )
        if folder:
            self.selected_folder = folder
            self.selected_files = []
            self.update_selection_display()
            # Use async loading for better performance
            self.load_photos_preview_async()

    def select_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Photo Files",
            "",
            "Image Files (*.jpg *.jpeg *.png *.bmp *.tiff *.webp);;All Files (*)",
        )
        if files:
            self.selected_files = files
            self.selected_folder = None
            self.update_selection_display()
            self.load_photos_preview_async()

    def update_selection_display(self):
        if self.selected_folder:
            # Show folder immediately, count images in background
            text = f"Folder: {self.selected_folder}\n(Scanning for images...)"
            self.selected_path_label.setText(text)

            # Start counting images asynchronously
            self.start_image_counting()
            return

        elif self.selected_files:
            count = len(self.selected_files)

            if self.current_mode == "single" and count == 1:
                text = f"Portrait: {os.path.basename(self.selected_files[0])}"
                self.visa_btn.setEnabled(True)
                self.enhance_btn.setEnabled(True)
            else:
                text = f"Selected {count} file{'s' if count != 1 else ''}\n"
                text += "\n".join([os.path.basename(f) for f in self.selected_files[:3]])
                if count > 3:
                    text += f"\n... and {count - 3} more"

                if self.current_mode == "batch":
                    # Enable step buttons when photos are available
                    self.step1_btn.setEnabled(count > 0)
                    self.step2_btn.setEnabled(count > 0)
                    self.step4_btn.setEnabled(count > 0)
                    self.update_step3_availability()
                elif self.current_mode == "single":
                    self.visa_btn.setEnabled(count == 1)
                    self.enhance_btn.setEnabled(count == 1)
                    if count > 1:
                        text += "\n⚠️ Single mode: Please select only one photo"
        else:
            if self.current_mode == "batch":
                text = "No sports photos selected"
                # Disable all step buttons when no photos selected
                self.step1_btn.setEnabled(False)
                self.step2_btn.setEnabled(False)
                self.step3_btn.setEnabled(False)
                self.step4_btn.setEnabled(False)
            else:
                text = "No portrait photo selected"
                self.visa_btn.setEnabled(False)
                self.enhance_btn.setEnabled(False)

        self.selected_path_label.setText(text)

    def update_step3_availability(self):
        """Update step 3 button availability based on completed steps."""
        has_quality = "quality" in self.completed_steps
        has_duplicates = "duplicates" in self.completed_steps
        has_photos = self.selected_folder or self.selected_files

        self.step3_btn.setEnabled(has_photos and has_quality and has_duplicates)

    def start_image_counting(self):
        """Start counting images in the selected folder asynchronously."""
        if not self.selected_folder:
            return

        # Stop any existing counting thread
        if self.image_counter_thread and self.image_counter_thread.isRunning():
            self.image_counter_thread.terminate()
            self.image_counter_thread.wait()

        # Start new counting thread
        self.image_counter_thread = ImageCounterThread(self.selected_folder)
        self.image_counter_thread.count_finished.connect(self.on_image_count_finished)
        self.image_counter_thread.start()

    def on_image_count_finished(self, image_count: int, preview_paths: List[str]):
        """Handle completion of image counting."""
        if not self.selected_folder:  # User might have changed selection
            return

        # Update display with actual count
        text = f"Folder: {self.selected_folder}\n({image_count} images found)"
        self.selected_path_label.setText(text)

        if self.current_mode == "batch":
            # Enable step buttons when photos are available
            self.step1_btn.setEnabled(image_count > 0)
            self.step2_btn.setEnabled(image_count > 0)
            self.step4_btn.setEnabled(image_count > 0)
            self.update_step3_availability()

        # Load preview with first few photos for quick display
        if preview_paths:
            self.photo_viewer.load_photos_fast(preview_paths, total_count=image_count)

            # Show helpful message for large folders
            if image_count > 50:
                self.results_text.setText(
                    f"📂 Large folder detected ({image_count} images)\n\n"
                    f"✨ Showing preview of first {len(preview_paths)} photos for faster loading\n\n"
                    f"💡 Processing steps will work with all {image_count} images\n\n"
                    "📋 Click any step above to begin processing"
                )

    def load_photos_preview_async(self):
        """Load photos preview asynchronously."""
        if self.selected_folder:
            # Folder loading is handled by image counting
            return
        elif self.selected_files:
            # For selected files, load immediately (usually not too many)
            self.photo_viewer.load_photos(self.selected_files)

    def load_photos_preview(self):
        if self.selected_folder:
            from ..utils.image_utils import get_image_paths

            photos = get_image_paths(self.selected_folder)
        else:
            photos = self.selected_files
        self.photo_viewer.load_photos(photos)

    def open_visa_dialog(self):
        dialog = VisaPhotoDialog(self.processor, self)
        dialog.exec()

    def open_settings(self):
        dialog = SettingsDialog(self.config, self)
        if dialog.exec():
            self.processor = PhotoProcessor(self.config)
            self.log_message("Settings updated", "success")

    def show_about(self):
        QMessageBox.about(
            self,
            "About Photo AI",
            "<h3>Photo AI v1.0.0</h3>"
            "<p>Intelligent photo processing and analysis toolkit</p>"
            "<p>Built with PyQt6 and advanced machine learning</p>"
            "<p>© 2024 Photo AI Team</p>",
        )

    def show_progress(self, show: bool):
        self.progress_bar.setVisible(show)
        self.progress_label.setVisible(show)
        if not show:
            self.progress_bar.setValue(0)
            self.progress_label.setText("")

    def update_progress(self, value: int):
        self.progress_bar.setValue(value)

    def update_status(self, message: str):
        self.progress_label.setText(message)
        self.statusBar().showMessage(message)
        self.log_message(message, "info")

    def on_processing_finished(self, results):
        self.show_progress(False)
        self.show_cancel_button(False)
        self.enable_available_steps()
        self.display_results(results)
        self.statusBar().showMessage("Processing completed successfully")
        self.log_message("Processing completed successfully", "success")

    def on_player_grouping_finished(self, results):
        """Handle completion of player grouping."""
        self.show_progress(False)
        self.show_cancel_button(False)
        # Re-enable step buttons after player grouping completes
        self.enable_available_steps()
        self.display_player_grouping_results(results)
        self.statusBar().showMessage("Player grouping completed successfully")
        self.log_message("Player grouping completed successfully", "success")

    def on_processing_error(self, error_message: str):
        self.show_progress(False)
        self.show_cancel_button(False)
        self.enable_available_steps()
        self.reset_step_button_statuses()
        QMessageBox.critical(self, "Processing Error", f"An error occurred:\n\n{error_message}")
        self.statusBar().showMessage("Processing failed")
        self.log_message(f"Error: {error_message}", "error")

    def display_results(self, results):
        """Display processing results."""
        # Show detailed results in the Results tab
        if not results.get("success", False):
            error_html = f"""
            <div style='color: #ff6b6b; background: #2a1a1a; padding: 15px; border-radius: 8px; border-left: 4px solid #ff6b6b;'>
                <h3 style='margin: 0 0 10px 0;'>❌ Processing Failed</h3>
                <p style='margin: 0;'>{results.get('error', 'Unknown error')}</p>
            </div>
            """
            self.results_widget.setHtml(error_html)
            # Simple error message in left panel
            self.results_text.setText("❌ Processing failed. Check Results tab for details.")
            return

        # Build detailed HTML for Results tab
        html = f"""
        <div style='font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; line-height: 1.4;'>
            <div style='color: #51cf66; background: #1a2e1a; padding: 15px; border-radius: 8px; border-left: 4px solid #51cf66; margin-bottom: 20px;'>
                <h3 style='margin: 0 0 5px 0; font-size: 16px;'>✅ Processing Completed Successfully!</h3>
            </div>
            
            <div style='background: #2a2a2a; padding: 15px; border-radius: 8px; margin-bottom: 15px;'>
                <h4 style='margin: 0 0 10px 0; color: #74c0fc; font-size: 14px;'>📁 Source Information</h4>
                <p style='margin: 5px 0; color: #d0d7de;'><strong>Directory:</strong> {os.path.basename(results['input_dir'])}</p>
                <p style='margin: 5px 0; color: #d0d7de;'><strong>Total Images:</strong> {results['total_images']}</p>
            </div>
        """

        # Simple progress summary for left panel
        progress_text = f"✅ Processing completed!\n\n📷 {results['total_images']} images processed"

        if "stages" in results:
            stages = results["stages"]

            if "sharpness" in stages:
                s = stages["sharpness"]
                total = s["sharp"] + s["blurry"]
                sharp_pct = (s["sharp"] / total * 100) if total > 0 else 0
                progress_text += f"\n🔍 {s['sharp']} sharp images ({sharp_pct:.0f}%)"

                html += f"""
                <div style='background: #2a2a2a; padding: 15px; border-radius: 8px; margin-bottom: 15px;'>
                    <h4 style='margin: 0 0 10px 0; color: #74c0fc; font-size: 14px;'>🔍 Sharpness Analysis</h4>
                    <div style='display: flex; justify-content: space-between; margin: 8px 0;'>
                        <span style='color: #51cf66;'>✨ Sharp images:</span>
                        <span style='color: #d0d7de; font-weight: bold;'>{s['sharp']} ({sharp_pct:.1f}%)</span>
                    </div>
                    <div style='display: flex; justify-content: space-between; margin: 8px 0;'>
                        <span style='color: #ff8cc8;'>🌫️ Blurry images:</span>
                        <span style='color: #d0d7de; font-weight: bold;'>{s['blurry']}</span>
                    </div>
                </div>
                """

            if "duplicates" in stages and "skipped" not in stages["duplicates"]:
                d = stages["duplicates"]["stats"]
                progress_text += f"\n🔄 {d['exact_duplicates_count']} duplicates removed"

                html += f"""
                <div style='background: #2a2a2a; padding: 15px; border-radius: 8px; margin-bottom: 15px;'>
                    <h4 style='margin: 0 0 10px 0; color: #74c0fc; font-size: 14px;'>🔄 Duplicate Detection</h4>
                    <div style='display: flex; justify-content: space-between; margin: 8px 0;'>
                        <span style='color: #ffd43b;'>🔗 Exact duplicates:</span>
                        <span style='color: #d0d7de; font-weight: bold;'>{d['exact_duplicates_count']}</span>
                    </div>
                    <div style='display: flex; justify-content: space-between; margin: 8px 0;'>
                        <span style='color: #74c0fc;'>🎭 Similar images:</span>
                        <span style='color: #d0d7de; font-weight: bold;'>{d['similar_images_count']}</span>
                    </div>
                    <div style='display: flex; justify-content: space-between; margin: 8px 0;'>
                        <span style='color: #51cf66;'>✨ Unique images:</span>
                        <span style='color: #d0d7de; font-weight: bold;'>{d['unique_images_estimate']}</span>
                    </div>
                </div>
                """

            if "selection" in stages and "skipped" not in stages["selection"]:
                s = stages["selection"]
                progress_text += f"\n⭐ {s['best_photos']} best photos selected"

                html += f"""
                <div style='background: #2a2a2a; padding: 15px; border-radius: 8px; margin-bottom: 15px;'>
                    <h4 style='margin: 0 0 10px 0; color: #74c0fc; font-size: 14px;'>⭐ Best Photo Selection</h4>
                    <div style='display: flex; justify-content: space-between; margin: 8px 0;'>
                        <span style='color: #ffd43b;'>🏆 Best photos selected:</span>
                        <span style='color: #d0d7de; font-weight: bold;'>{s['best_photos']}</span>
                    </div>
                </div>
                """

        html += f"""
            <div style='background: #1a2e2e; padding: 12px; border-radius: 6px; border-left: 3px solid #74c0fc; margin-top: 20px;'>
                <p style='margin: 0; color: #74c0fc; font-size: 13px;'>💡 <em>Check the Log tab for detailed processing information</em></p>
            </div>
        </div>
        """

        # Set detailed results in Results tab
        self.results_widget.setHtml(html)

        # Set simple progress summary in left panel
        progress_text += "\n\n📊 See Results tab for full details"
        self.results_text.setText(progress_text)

        # Switch to Results tab to show the results
        self.tab_widget.setCurrentIndex(1)

    def display_analysis_results(self, results):
        """Display analysis results."""
        html = f"""
        <div style='font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; line-height: 1.4;'>
            <div style='color: #74c0fc; background: #1a1a2e; padding: 15px; border-radius: 8px; border-left: 4px solid #74c0fc; margin-bottom: 20px;'>
                <h3 style='margin: 0 0 5px 0; font-size: 16px;'>🔍 Analysis Completed!</h3>
            </div>
        """

        # Simple progress summary for left panel
        progress_text = "🔍 Analysis completed!\n"

        if "sharpness" in results:
            sharp_count = sum(
                1 for r in results["sharpness"].values() if r.get("overall_is_sharp", False)
            )
            total = len(results["sharpness"])
            sharp_pct = (sharp_count / total * 100) if total > 0 else 0
            progress_text += f"\n📸 {sharp_count}/{total} sharp images ({sharp_pct:.0f}%)"

            html += f"""
            <div style='background: #2a2a2a; padding: 15px; border-radius: 8px; margin-bottom: 15px;'>
                <h4 style='margin: 0 0 10px 0; color: #74c0fc; font-size: 14px;'>📸 Sharpness Analysis</h4>
                <div style='display: flex; justify-content: space-between; margin: 8px 0;'>
                    <span style='color: #51cf66;'>✨ Sharp images:</span>
                    <span style='color: #d0d7de; font-weight: bold;'>{sharp_count}/{total} ({sharp_pct:.1f}%)</span>
                </div>
            </div>
            """

        if "duplicates" in results:
            stats = results["duplicates"]["stats"]
            progress_text += f"\n🔄 {stats['exact_duplicates_count']} duplicates found"

            html += f"""
            <div style='background: #2a2a2a; padding: 15px; border-radius: 8px; margin-bottom: 15px;'>
                <h4 style='margin: 0 0 10px 0; color: #74c0fc; font-size: 14px;'>🔄 Duplicate Analysis</h4>
                <div style='display: flex; justify-content: space-between; margin: 8px 0;'>
                    <span style='color: #ffd43b;'>🔗 Exact duplicates:</span>
                    <span style='color: #d0d7de; font-weight: bold;'>{stats['exact_duplicates_count']}</span>
                </div>
                <div style='display: flex; justify-content: space-between; margin: 8px 0;'>
                    <span style='color: #74c0fc;'>🎭 Similar images:</span>
                    <span style='color: #d0d7de; font-weight: bold;'>{stats['similar_images_count']}</span>
                </div>
            </div>
            """

        if "faces" in results:
            face_count = sum(1 for r in results["faces"].values() if r.get("face_info"))
            total = len(results["faces"])
            face_pct = (face_count / total * 100) if total > 0 else 0
            progress_text += f"\n👤 {face_count}/{total} images with faces ({face_pct:.0f}%)"

            html += f"""
            <div style='background: #2a2a2a; padding: 15px; border-radius: 8px; margin-bottom: 15px;'>
                <h4 style='margin: 0 0 10px 0; color: #74c0fc; font-size: 14px;'>👤 Face Detection</h4>
                <div style='display: flex; justify-content: space-between; margin: 8px 0;'>
                    <span style='color: #ff8cc8;'>😊 Images with faces:</span>
                    <span style='color: #d0d7de; font-weight: bold;'>{face_count}/{total} ({face_pct:.1f}%)</span>
                </div>
            </div>
            """

        html += f"""
            <div style='background: #1a2e2e; padding: 12px; border-radius: 6px; border-left: 3px solid #74c0fc; margin-top: 20px;'>
                <p style='margin: 0; color: #74c0fc; font-size: 13px;'>💡 <em>Check the Log tab for detailed analysis information</em></p>
            </div>
        </div>
        """

        # Set detailed results in Results tab
        self.results_widget.setHtml(html)

        # Set simple progress summary in left panel
        progress_text += "\n\n📊 See Results tab for full details"
        self.results_text.setText(progress_text)

        # Switch to Results tab to show the results
        self.tab_widget.setCurrentIndex(1)

    def display_player_grouping_results(self, results):
        """Display player grouping results."""
        if not results.get("success", False):
            error_html = f"""
            <div style='color: #ff6b6b; background: #2a1a1a; padding: 15px; border-radius: 8px; border-left: 4px solid #ff6b6b;'>
                <h3 style='margin: 0 0 10px 0;'>❌ Player Grouping Failed</h3>
                <p style='margin: 0;'>{results.get('error', 'Unknown error')}</p>
            </div>
            """
            self.results_widget.setHtml(error_html)
            self.results_text.setText("❌ Player grouping failed. Check Results tab for details.")
            return

        # Build detailed HTML for Results tab
        html = f"""
        <div style='font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; line-height: 1.4;'>
            <div style='color: #74c0fc; background: #1a1a2e; padding: 15px; border-radius: 8px; border-left: 4px solid #74c0fc; margin-bottom: 20px;'>
                <h3 style='margin: 0 0 5px 0; font-size: 16px;'>👥 Player Grouping Completed!</h3>
            </div>
            
            <div style='background: #2a2a2a; padding: 15px; border-radius: 8px; margin-bottom: 15px;'>
                <h4 style='margin: 0 0 10px 0; color: #74c0fc; font-size: 14px;'>📊 Grouping Statistics</h4>
                <div style='display: flex; justify-content: space-between; margin: 8px 0;'>
                    <span style='color: #d0d7de;'>Total Photos:</span>
                    <span style='color: #d0d7de; font-weight: bold;'>{results.get('total_photos', 0)}</span>
                </div>
                <div style='display: flex; justify-content: space-between; margin: 8px 0;'>
                    <span style='color: #51cf66;'>Photos with Faces:</span>
                    <span style='color: #d0d7de; font-weight: bold;'>{results.get('photos_with_faces', 0)}</span>
                </div>
                <div style='display: flex; justify-content: space-between; margin: 8px 0;'>
                    <span style='color: #ffd43b;'>Player Groups Created:</span>
                    <span style='color: #d0d7de; font-weight: bold;'>{results.get('player_groups', 0)}</span>
                </div>
                <div style='display: flex; justify-content: space-between; margin: 8px 0;'>
                    <span style='color: #ff8cc8;'>Ungrouped Photos:</span>
                    <span style='color: #d0d7de; font-weight: bold;'>{results.get('ungrouped_photos', 0)}</span>
                </div>
            </div>
        """

        # Add individual group details
        if results.get("group_stats"):
            html += """
            <div style='background: #2a2a2a; padding: 15px; border-radius: 8px; margin-bottom: 15px;'>
                <h4 style='margin: 0 0 10px 0; color: #74c0fc; font-size: 14px;'>👤 Individual Player Groups</h4>
            """

            for group_id, stats in results["group_stats"].items():
                confidence_pct = stats["avg_confidence"] * 100
                html += f"""
                <div style='display: flex; justify-content: space-between; margin: 8px 0; padding: 5px; background: #1a1a1a; border-radius: 4px;'>
                    <span style='color: #74c0fc;'>Player {group_id}:</span>
                    <span style='color: #d0d7de;'>{stats['photo_count']} photos (confidence: {confidence_pct:.1f}%)</span>
                </div>
                """

            html += "</div>"

        if results.get("output_directory"):
            html += f"""
            <div style='background: #1a2e1a; padding: 15px; border-radius: 8px; margin-bottom: 15px;'>
                <h4 style='margin: 0 0 10px 0; color: #51cf66; font-size: 14px;'>📁 Output Location</h4>
                <p style='margin: 5px 0; color: #d0d7de; word-break: break-all;'>{results['output_directory']}</p>
            </div>
            """

        html += f"""
            <div style='background: #1a2e2e; padding: 12px; border-radius: 6px; border-left: 3px solid #74c0fc; margin-top: 20px;'>
                <p style='margin: 0; color: #74c0fc; font-size: 13px;'>💡 <em>Check the Log tab for detailed processing information</em></p>
            </div>
        </div>
        """

        # Set detailed results in Results tab
        self.results_widget.setHtml(html)

        # Simple summary for left panel
        total_photos = results.get("total_photos", 0)
        player_groups = results.get("player_groups", 0)
        photos_with_faces = results.get("photos_with_faces", 0)

        progress_text = f"👥 Player grouping completed!\n\n📷 {total_photos} photos analyzed"
        progress_text += f"\n👤 {photos_with_faces} photos with faces"
        progress_text += f"\n🎯 {player_groups} player groups created"
        progress_text += "\n\n📊 See Results tab for full details"

        self.results_text.setText(progress_text)

        # Switch to Results tab to show the results
        self.tab_widget.setCurrentIndex(1)

    def log_message(self, message: str, level: str = "info"):
        from datetime import datetime

        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {message}"

        if hasattr(self, "logger"):
            if level == "info":
                self.logger.log_info(formatted_msg)
            elif level == "success":
                self.logger.log_success(formatted_msg)
            elif level == "warning":
                self.logger.log_warning(formatted_msg)
            elif level == "error":
                self.logger.log_error(formatted_msg)
            else:
                self.logger.log(formatted_msg)

    def closeEvent(self, event):
        if self.current_processing_thread and self.current_processing_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Processing in Progress",
                "Photo processing is still running. Are you sure you want to exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.current_processing_thread.terminate()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
