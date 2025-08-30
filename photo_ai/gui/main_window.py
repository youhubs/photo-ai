"""Main window for Photo AI desktop application."""

import os
from typing import Optional, List
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QTextEdit, QProgressBar, QGroupBox,
    QFileDialog, QMessageBox, QTabWidget, QListWidget, QListWidgetItem,
    QSplitter, QFrame, QScrollArea
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPixmap, QIcon

from ..core.photo_processor import PhotoProcessor
from ..core.config import Config
from .dialogs.visa_dialog import VisaPhotoDialog
from .dialogs.settings_dialog import SettingsDialog
from .widgets.photo_viewer import PhotoViewer
from .widgets.processing_thread import ProcessingThread


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
        
        # Create status bar
        self.statusBar().showMessage("Ready to process photos")
        
        # Create menu bar
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
        
        # Folder selection
        self.select_folder_btn = QPushButton("📁 Select Photo Folder")
        self.select_folder_btn.setMinimumHeight(50)
        self.select_folder_btn.setToolTip("Select a folder containing photos to process")
        selection_layout.addWidget(self.select_folder_btn)
        
        # Individual file selection
        self.select_files_btn = QPushButton("🖼️ Select Individual Photos")
        self.select_files_btn.setMinimumHeight(50)
        self.select_files_btn.setToolTip("Select specific photo files to process")
        selection_layout.addWidget(self.select_files_btn)
        
        # Selected path display
        self.selected_path_label = QLabel("No folder or files selected")
        self.selected_path_label.setWordWrap(True)
        self.selected_path_label.setStyleSheet("color: #666666; padding: 10px; border: 1px solid #444444; border-radius: 5px;")
        selection_layout.addWidget(self.selected_path_label)
        
        layout.addWidget(selection_group)
        
        # Processing options group
        processing_group = QGroupBox("Processing Options")
        processing_layout = QVBoxLayout(processing_group)
        
        # Main processing button
        self.process_btn = QPushButton("🚀 Process Photos")
        self.process_btn.setMinimumHeight(50)
        self.process_btn.setEnabled(False)
        self.process_btn.setToolTip("Run the complete photo processing pipeline")
        processing_layout.addWidget(self.process_btn)
        
        # Visa photo button
        self.visa_btn = QPushButton("📄 Create Visa Photo")
        self.visa_btn.setMinimumHeight(40)
        self.visa_btn.setToolTip("Create a visa/passport photo from selected image")
        processing_layout.addWidget(self.visa_btn)
        
        # Analyze only button
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
        """Create the right panel with tabs for different views."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Photo viewer tab
        self.photo_viewer = PhotoViewer()
        self.tab_widget.addTab(self.photo_viewer, "📷 Photo Viewer")
        
        # Results tab
        self.results_list = QListWidget()
        self.tab_widget.addTab(self.results_list, "📊 Results")
        
        # Log tab
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 10))
        self.tab_widget.addTab(self.log_text, "📝 Log")
        
        return panel
    
    def create_menu_bar(self):
        """Create the application menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        file_menu.addAction("Select Folder", self.select_folder)
        file_menu.addAction("Select Files", self.select_files)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        tools_menu.addAction("Create Visa Photo", self.open_visa_dialog)
        tools_menu.addAction("Settings", self.open_settings)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        help_menu.addAction("About", self.show_about)
        
    def setup_connections(self):
        """Setup signal connections."""
        self.select_folder_btn.clicked.connect(self.select_folder)
        self.select_files_btn.clicked.connect(self.select_files)
        self.process_btn.clicked.connect(self.start_processing)
        self.visa_btn.clicked.connect(self.open_visa_dialog)
        self.analyze_btn.clicked.connect(self.start_analysis)
        self.settings_btn.clicked.connect(self.open_settings)
        
    def select_folder(self):
        """Select a folder containing photos."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Photo Folder",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        
        if folder:
            self.selected_folder = folder
            self.selected_files = []
            self.update_selection_display()
            self.load_photos_preview()
            
    def select_files(self):
        """Select individual photo files."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Photo Files",
            "",
            "Image Files (*.jpg *.jpeg *.png *.bmp *.tiff *.webp);;All Files (*)"
        )
        
        if files:
            self.selected_files = files
            self.selected_folder = None
            self.update_selection_display()
            self.load_photos_preview()
            
    def update_selection_display(self):
        """Update the selection display and enable/disable buttons."""
        if self.selected_folder:
            from ..utils.image_utils import get_image_paths
            image_count = len(get_image_paths(self.selected_folder))
            text = f"Folder: {self.selected_folder}\\n({image_count} images found)"
            self.process_btn.setEnabled(image_count > 0)
            self.analyze_btn.setEnabled(image_count > 0)
        elif self.selected_files:
            count = len(self.selected_files)
            text = f"Selected {count} file{'s' if count != 1 else ''}\\n"
            text += "\\n".join([os.path.basename(f) for f in self.selected_files[:3]])
            if count > 3:
                text += f"\\n... and {count - 3} more"
            self.process_btn.setEnabled(count > 0)
            self.analyze_btn.setEnabled(count > 0)
        else:
            text = "No folder or files selected"
            self.process_btn.setEnabled(False)
            self.analyze_btn.setEnabled(False)
            
        self.selected_path_label.setText(text)
        
    def load_photos_preview(self):
        """Load photo previews in the viewer."""
        if self.selected_folder:
            from ..utils.image_utils import get_image_paths
            photos = get_image_paths(self.selected_folder)
        else:
            photos = self.selected_files
            
        self.photo_viewer.load_photos(photos)
        
    def start_processing(self):
        """Start the photo processing in a separate thread."""
        if self.current_processing_thread and self.current_processing_thread.isRunning():
            return
            
        # Get photos to process
        if self.selected_folder:
            input_source = self.selected_folder
        else:
            input_source = self.selected_files
            
        # Create and start processing thread
        self.current_processing_thread = ProcessingThread(
            self.processor, input_source, "process"
        )
        
        # Connect signals
        self.current_processing_thread.progress_updated.connect(self.update_progress)
        self.current_processing_thread.status_updated.connect(self.update_status)
        self.current_processing_thread.finished_processing.connect(self.on_processing_finished)
        self.current_processing_thread.error_occurred.connect(self.on_processing_error)
        
        # Update UI
        self.show_progress(True)
        self.process_btn.setEnabled(False)
        self.analyze_btn.setEnabled(False)
        
        # Start processing
        self.current_processing_thread.start()
        self.log_message("Started photo processing...")
        
    def start_analysis(self):
        """Start photo analysis only."""
        if self.current_processing_thread and self.current_processing_thread.isRunning():
            return
            
        # Get photos to analyze
        if self.selected_folder:
            input_source = self.selected_folder
        else:
            input_source = self.selected_files
            
        # Create and start analysis thread
        self.current_processing_thread = ProcessingThread(
            self.processor, input_source, "analyze"
        )
        
        # Connect signals
        self.current_processing_thread.progress_updated.connect(self.update_progress)
        self.current_processing_thread.status_updated.connect(self.update_status)
        self.current_processing_thread.finished_processing.connect(self.on_analysis_finished)
        self.current_processing_thread.error_occurred.connect(self.on_processing_error)
        
        # Update UI
        self.show_progress(True)
        self.process_btn.setEnabled(False)
        self.analyze_btn.setEnabled(False)
        
        # Start analysis
        self.current_processing_thread.start()
        self.log_message("Started photo analysis...")
        
    def open_visa_dialog(self):
        """Open the visa photo creation dialog."""
        dialog = VisaPhotoDialog(self.processor, self)
        dialog.exec()
        
    def open_settings(self):
        """Open the settings dialog."""
        dialog = SettingsDialog(self.config, self)
        if dialog.exec():
            # Reload processor with new config
            self.processor = PhotoProcessor(self.config)
            self.log_message("Settings updated")
            
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Photo AI",
            "<h3>Photo AI v1.0.0</h3>"
            "<p>Intelligent photo processing and analysis toolkit</p>"
            "<p>Built with PyQt6 and advanced machine learning</p>"
            "<p>© 2024 Photo AI Team</p>"
        )
        
    def show_progress(self, show: bool):
        """Show or hide progress indicators."""
        self.progress_bar.setVisible(show)
        self.progress_label.setVisible(show)
        if not show:
            self.progress_bar.setValue(0)
            self.progress_label.setText("")
            
    def update_progress(self, value: int):
        """Update progress bar value."""
        self.progress_bar.setValue(value)
        
    def update_status(self, message: str):
        """Update status message."""
        self.progress_label.setText(message)
        self.statusBar().showMessage(message)
        self.log_message(message)
        
    def on_processing_finished(self, results):
        """Handle processing completion."""
        self.show_progress(False)
        self.process_btn.setEnabled(True)
        self.analyze_btn.setEnabled(True)
        
        # Display results
        self.display_results(results)
        self.statusBar().showMessage("Processing completed successfully")
        self.log_message("Processing completed successfully")
        
    def on_analysis_finished(self, results):
        """Handle analysis completion."""
        self.show_progress(False)
        self.process_btn.setEnabled(True)
        self.analyze_btn.setEnabled(True)
        
        # Display analysis results
        self.display_analysis_results(results)
        self.statusBar().showMessage("Analysis completed successfully")
        self.log_message("Analysis completed successfully")
        
    def on_processing_error(self, error_message: str):
        """Handle processing error."""
        self.show_progress(False)
        self.process_btn.setEnabled(True)
        self.analyze_btn.setEnabled(True)
        
        QMessageBox.critical(self, "Processing Error", f"An error occurred:\\n\\n{error_message}")
        self.statusBar().showMessage("Processing failed")
        self.log_message(f"Error: {error_message}")
        
    def display_results(self, results):
        """Display processing results."""
        if not results.get('success', False):
            self.results_text.setText(f"Processing failed: {results.get('error', 'Unknown error')}")
            return
            
        text = f"✅ Processing completed successfully!\\n\\n"
        text += f"📁 Input directory: {results['input_dir']}\\n"
        text += f"📷 Total images: {results['total_images']}\\n\\n"
        
        if 'stages' in results:
            stages = results['stages']
            
            if 'sharpness' in stages:
                s = stages['sharpness']
                text += f"🔍 Sharpness Analysis:\\n"
                text += f"   • Sharp images: {s['sharp']}\\n"
                text += f"   • Blurry images: {s['blurry']}\\n\\n"
                
            if 'duplicates' in stages and 'skipped' not in stages['duplicates']:
                d = stages['duplicates']['stats']
                text += f"🔄 Duplicate Detection:\\n"
                text += f"   • Exact duplicates: {d['exact_duplicates_count']}\\n"
                text += f"   • Similar images: {d['similar_images_count']}\\n"
                text += f"   • Unique images: {d['unique_images_estimate']}\\n\\n"
                
            if 'selection' in stages and 'skipped' not in stages['selection']:
                s = stages['selection']
                text += f"⭐ Best Photo Selection:\\n"
                text += f"   • Best photos selected: {s['best_photos']}\\n\\n"
        
        text += f"📊 Check the Results tab for detailed information."
        self.results_text.setText(text)
        
    def display_analysis_results(self, results):
        """Display analysis results."""
        text = f"🔍 Analysis completed!\\n\\n"
        
        if 'sharpness' in results:
            sharp_count = sum(1 for r in results['sharpness'].values() 
                            if r.get('overall_is_sharp', False))
            total = len(results['sharpness'])
            text += f"📸 Sharpness: {sharp_count}/{total} images are sharp\\n"
            
        if 'duplicates' in results:
            stats = results['duplicates']['stats']
            text += f"🔄 Duplicates: {stats['exact_duplicates_count']} exact duplicates found\\n"
            text += f"🎭 Similar: {stats['similar_images_count']} similar images found\\n"
            
        if 'faces' in results:
            face_count = sum(1 for r in results['faces'].values() 
                           if r.get('face_info'))
            total = len(results['faces'])
            text += f"👤 Faces: {face_count}/{total} images have detectable faces\\n"
            
        self.results_text.setText(text)
        
    def log_message(self, message: str):
        """Add message to log tab."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        
    def closeEvent(self, event):
        """Handle application close event."""
        if self.current_processing_thread and self.current_processing_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Processing in Progress",
                "Photo processing is still running. Are you sure you want to exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.current_processing_thread.terminate()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()