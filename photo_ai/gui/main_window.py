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
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from ..core.photo_processor import PhotoProcessor
from ..core.config import Config
from .dialogs.visa_dialog import VisaPhotoDialog
from .dialogs.settings_dialog import SettingsDialog
from .widgets.photo_viewer import PhotoViewer
from .widgets.processing_thread import ProcessingThread
from .widgets.logger_widget import LoggerWidget


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
        subtitle.setStyleSheet("color: #888888; margin-bottom: 20px;")
        layout.addWidget(subtitle)

        # Photo selection group
        selection_group = QGroupBox("Photo Selection")
        selection_layout = QVBoxLayout(selection_group)

        self.select_folder_btn = QPushButton("📁 Select Photo Folder")
        self.select_folder_btn.setMinimumHeight(50)
        self.select_folder_btn.setToolTip("Select a folder containing photos to process")
        selection_layout.addWidget(self.select_folder_btn)

        self.select_files_btn = QPushButton("🖼️ Select Individual Photos")
        self.select_files_btn.setMinimumHeight(50)
        self.select_files_btn.setToolTip("Select specific photo files to process")
        selection_layout.addWidget(self.select_files_btn)

        self.selected_path_label = QLabel("No folder or files selected")
        self.selected_path_label.setWordWrap(True)
        self.selected_path_label.setStyleSheet(
            "color: #666666; padding: 10px; border: 1px solid #444444; border-radius: 5px;"
        )
        selection_layout.addWidget(self.selected_path_label)

        layout.addWidget(selection_group)

        # Processing options
        processing_group = QGroupBox("Processing Options")
        processing_layout = QVBoxLayout(processing_group)

        self.process_btn = QPushButton("🚀 Process Photos")
        self.process_btn.setMinimumHeight(50)
        self.process_btn.setEnabled(False)
        self.process_btn.setToolTip("Run the complete photo processing pipeline")
        processing_layout.addWidget(self.process_btn)

        self.visa_btn = QPushButton("📄 Create Visa Photo")
        self.visa_btn.setMinimumHeight(40)
        self.visa_btn.setToolTip("Create a visa/passport photo from selected image")
        processing_layout.addWidget(self.visa_btn)

        self.analyze_btn = QPushButton("🔍 Analyze Quality Only")
        self.analyze_btn.setMinimumHeight(40)
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setToolTip("Analyze photo quality without processing")
        processing_layout.addWidget(self.analyze_btn)

        layout.addWidget(processing_group)

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
        self.select_folder_btn.clicked.connect(self.select_folder)
        self.select_files_btn.clicked.connect(self.select_files)
        self.process_btn.clicked.connect(self.start_processing)
        self.visa_btn.clicked.connect(self.open_visa_dialog)
        self.analyze_btn.clicked.connect(self.start_analysis)
        self.settings_btn.clicked.connect(self.open_settings)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Photo Folder", "", QFileDialog.Option.ShowDirsOnly
        )
        if folder:
            self.selected_folder = folder
            self.selected_files = []
            self.update_selection_display()
            self.load_photos_preview()

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
            self.load_photos_preview()

    def update_selection_display(self):
        if self.selected_folder:
            from ..utils.image_utils import get_image_paths
            image_count = len(get_image_paths(self.selected_folder))
            text = f"Folder: {self.selected_folder}\n({image_count} images found)"
            self.process_btn.setEnabled(image_count > 0)
            self.analyze_btn.setEnabled(image_count > 0)
        elif self.selected_files:
            count = len(self.selected_files)
            text = f"Selected {count} file{'s' if count != 1 else ''}\n"
            text += "\n".join([os.path.basename(f) for f in self.selected_files[:3]])
            if count > 3:
                text += f"\n... and {count - 3} more"
            self.process_btn.setEnabled(count > 0)
            self.analyze_btn.setEnabled(count > 0)
        else:
            text = "No folder or files selected"
            self.process_btn.setEnabled(False)
            self.analyze_btn.setEnabled(False)
        self.selected_path_label.setText(text)

    def load_photos_preview(self):
        if self.selected_folder:
            from ..utils.image_utils import get_image_paths
            photos = get_image_paths(self.selected_folder)
        else:
            photos = self.selected_files
        self.photo_viewer.load_photos(photos)

    def start_processing(self):
        if self.current_processing_thread and self.current_processing_thread.isRunning():
            return

        input_source = self.selected_folder if self.selected_folder else self.selected_files
        self.current_processing_thread = ProcessingThread(self.processor, input_source, "process")
        self.current_processing_thread.progress_updated.connect(self.update_progress)
        self.current_processing_thread.status_updated.connect(self.update_status)
        self.current_processing_thread.finished_processing.connect(self.on_processing_finished)
        self.current_processing_thread.error_occurred.connect(self.on_processing_error)

        self.show_progress(True)
        self.process_btn.setEnabled(False)
        self.analyze_btn.setEnabled(False)

        self.current_processing_thread.start()
        self.log_message("Started photo processing...", "info")

    def start_analysis(self):
        if self.current_processing_thread and self.current_processing_thread.isRunning():
            return

        input_source = self.selected_folder if self.selected_folder else self.selected_files
        self.current_processing_thread = ProcessingThread(self.processor, input_source, "analyze")
        self.current_processing_thread.progress_updated.connect(self.update_progress)
        self.current_processing_thread.status_updated.connect(self.update_status)
        self.current_processing_thread.finished_processing.connect(self.on_analysis_finished)
        self.current_processing_thread.error_occurred.connect(self.on_processing_error)

        self.show_progress(True)
        self.process_btn.setEnabled(False)
        self.analyze_btn.setEnabled(False)

        self.current_processing_thread.start()
        self.log_message("Started photo analysis...", "info")

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
        self.process_btn.setEnabled(True)
        self.analyze_btn.setEnabled(True)
        self.display_results(results)
        self.statusBar().showMessage("Processing completed successfully")
        self.log_message("Processing completed successfully", "success")

    def on_analysis_finished(self, results):
        self.show_progress(False)
        self.process_btn.setEnabled(True)
        self.analyze_btn.setEnabled(True)
        self.display_analysis_results(results)
        self.statusBar().showMessage("Analysis completed successfully")
        self.log_message("Analysis completed successfully", "success")

    def on_processing_error(self, error_message: str):
        self.show_progress(False)
        self.process_btn.setEnabled(True)
        self.analyze_btn.setEnabled(True)
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
                total = s['sharp'] + s['blurry']
                sharp_pct = (s['sharp'] / total * 100) if total > 0 else 0
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
