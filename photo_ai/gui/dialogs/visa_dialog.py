"""Dialog for creating visa photos."""

import os
from typing import Optional
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QFileDialog, QProgressBar, QTextEdit, QGroupBox, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QMessageBox, QFrame, QWidget
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QFont

from ...core.photo_processor import PhotoProcessor
from ..widgets.processing_thread import ProcessingThread


class VisaPhotoDialog(QDialog):
    """Dialog for creating visa/passport photos."""
    
    def __init__(self, processor: PhotoProcessor, parent=None):
        super().__init__(parent)
        self.processor = processor
        self.input_file: Optional[str] = None
        self.output_file: Optional[str] = None
        self.processing_thread: Optional[ProcessingThread] = None
        
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Setup the dialog UI."""
        self.setWindowTitle("Create Visa Photo")
        self.setModal(True)
        self.resize(800, 600)
        
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Visa Photo Creator")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("margin: 10px; color: #2196F3;")
        layout.addWidget(title)
        
        # Description
        desc = QLabel("Create compliant visa/passport photos with precise dimensions and background removal.")
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setStyleSheet("color: #666666; margin-bottom: 20px;")
        layout.addWidget(desc)
        
        # Main content layout
        content_layout = QHBoxLayout()
        layout.addLayout(content_layout)
        
        # Left panel - Controls
        left_panel = self.create_left_panel()
        content_layout.addWidget(left_panel)
        
        # Right panel - Preview
        right_panel = self.create_right_panel()
        content_layout.addWidget(right_panel)
        
        # Progress section
        self.progress_group = QGroupBox("Processing Progress")
        progress_layout = QVBoxLayout(self.progress_group)
        
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("")
        progress_layout.addWidget(self.progress_label)
        
        self.progress_group.setVisible(False)
        layout.addWidget(self.progress_group)
        
        # Button layout
        button_layout = QHBoxLayout()
        
        self.process_btn = QPushButton("🚀 Create Visa Photo")
        self.process_btn.setMinimumHeight(40)
        self.process_btn.setEnabled(False)
        button_layout.addWidget(self.process_btn)
        
        self.close_btn = QPushButton("Close")
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
    def create_left_panel(self) -> QWidget:
        """Create the left control panel."""
        panel = QFrame()
        panel.setMaximumWidth(350)
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(panel)
        
        # File selection group
        file_group = QGroupBox("Input Image")
        file_layout = QVBoxLayout(file_group)
        
        self.select_input_btn = QPushButton("📁 Select Photo")
        self.select_input_btn.setMinimumHeight(40)
        file_layout.addWidget(self.select_input_btn)
        
        self.input_file_label = QLabel("No file selected")
        self.input_file_label.setWordWrap(True)
        self.input_file_label.setStyleSheet("color: #666666; padding: 10px; border: 1px solid #444444; border-radius: 5px;")
        file_layout.addWidget(self.input_file_label)
        
        layout.addWidget(file_group)
        
        # Settings group
        settings_group = QGroupBox("Photo Settings")
        settings_layout = QGridLayout(settings_group)
        
        # Dimensions
        settings_layout.addWidget(QLabel("Width (mm):"), 0, 0)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(20, 100)
        self.width_spin.setValue(self.processor.config.visa.photo_width_mm)
        settings_layout.addWidget(self.width_spin, 0, 1)
        
        settings_layout.addWidget(QLabel("Height (mm):"), 1, 0)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(20, 150)
        self.height_spin.setValue(self.processor.config.visa.photo_height_mm)
        settings_layout.addWidget(self.height_spin, 1, 1)
        
        # DPI
        settings_layout.addWidget(QLabel("DPI:"), 2, 0)
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(150, 600)
        self.dpi_spin.setValue(self.processor.config.visa.dpi)
        settings_layout.addWidget(self.dpi_spin, 2, 1)
        
        # Face size ratio
        settings_layout.addWidget(QLabel("Face Size Ratio:"), 3, 0)
        self.face_ratio_spin = QDoubleSpinBox()
        self.face_ratio_spin.setRange(0.3, 0.7)
        self.face_ratio_spin.setSingleStep(0.05)
        self.face_ratio_spin.setValue(self.processor.config.visa.face_height_ratio)
        settings_layout.addWidget(self.face_ratio_spin, 3, 1)
        
        layout.addWidget(settings_group)
        
        # Output settings group
        output_group = QGroupBox("Output Settings")
        output_layout = QVBoxLayout(output_group)
        
        self.select_output_btn = QPushButton("💾 Select Output Location")
        output_layout.addWidget(self.select_output_btn)
        
        self.output_file_label = QLabel("Auto-generated filename will be used")
        self.output_file_label.setWordWrap(True)
        self.output_file_label.setStyleSheet("color: #666666; padding: 5px;")
        output_layout.addWidget(self.output_file_label)
        
        self.debug_check = QCheckBox("Enable debug mode (save intermediate steps)")
        output_layout.addWidget(self.debug_check)
        
        layout.addWidget(output_group)
        
        # Requirements info
        req_group = QGroupBox("Requirements")
        req_layout = QVBoxLayout(req_group)
        
        requirements_text = """
• Photo must show a clear frontal face
• Face should occupy 30-40% of image height
• Minimum resolution: 1600×1200 pixels
• Good lighting without shadows
• Plain background (will be removed)
        """.strip()
        
        req_label = QLabel(requirements_text)
        req_label.setWordWrap(True)
        req_label.setStyleSheet("color: #888888; font-size: 11px;")
        req_layout.addWidget(req_label)
        
        layout.addWidget(req_group)
        
        layout.addStretch()
        return panel
        
    def create_right_panel(self) -> QWidget:
        """Create the right preview panel."""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(panel)
        
        # Preview title
        preview_title = QLabel("Preview")
        preview_title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        preview_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(preview_title)
        
        # Preview image
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(300, 400)
        self.preview_label.setStyleSheet("""
            QLabel {
                background-color: #2a2a2a;
                border: 2px dashed #555555;
                border-radius: 10px;
                color: #888888;
                font-size: 14px;
            }
        """)
        self.preview_label.setText("Select a photo to see preview")
        layout.addWidget(self.preview_label)
        
        # Results area
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(150)
        self.results_text.setPlaceholderText("Processing results will appear here...")
        layout.addWidget(self.results_text)
        
        return panel
        
    def setup_connections(self):
        """Setup signal connections."""
        self.select_input_btn.clicked.connect(self.select_input_file)
        self.select_output_btn.clicked.connect(self.select_output_file)
        self.process_btn.clicked.connect(self.create_visa_photo)
        self.close_btn.clicked.connect(self.close)
        
        # Connect settings changes to update preview
        self.width_spin.valueChanged.connect(self.update_settings)
        self.height_spin.valueChanged.connect(self.update_settings)
        self.dpi_spin.valueChanged.connect(self.update_settings)
        self.face_ratio_spin.valueChanged.connect(self.update_settings)
        
    def select_input_file(self):
        """Select input photo file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Photo for Visa Processing",
            "",
            "Image Files (*.jpg *.jpeg *.png *.bmp);;All Files (*)"
        )
        
        if file_path:
            self.input_file = file_path
            filename = os.path.basename(file_path)
            self.input_file_label.setText(f"Selected: {filename}")
            self.process_btn.setEnabled(True)
            self.load_preview()
            
    def select_output_file(self):
        """Select output file location."""
        if not self.input_file:
            QMessageBox.warning(self, "Warning", "Please select an input file first.")
            return
            
        # Default filename based on input
        input_name = os.path.splitext(os.path.basename(self.input_file))[0]
        default_name = f"{input_name}_visa.jpg"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Visa Photo As",
            default_name,
            "JPEG Files (*.jpg);;All Files (*)"
        )
        
        if file_path:
            self.output_file = file_path
            self.output_file_label.setText(f"Output: {os.path.basename(file_path)}")
            
    def load_preview(self):
        """Load preview of the input image."""
        if not self.input_file:
            return
            
        try:
            pixmap = QPixmap(self.input_file)
            if not pixmap.isNull():
                # Scale to fit preview area
                scaled_pixmap = pixmap.scaled(
                    300, 400,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.preview_label.setPixmap(scaled_pixmap)
            else:
                self.preview_label.setText("Failed to load image preview")
        except Exception as e:
            self.preview_label.setText(f"Preview error: {str(e)}")
            
    def update_settings(self):
        """Update processor settings from UI controls."""
        self.processor.config.visa.photo_width_mm = self.width_spin.value()
        self.processor.config.visa.photo_height_mm = self.height_spin.value()
        self.processor.config.visa.dpi = self.dpi_spin.value()
        self.processor.config.visa.face_height_ratio = self.face_ratio_spin.value()
        
    def create_visa_photo(self):
        """Create the visa photo."""
        if not self.input_file:
            QMessageBox.warning(self, "Warning", "Please select an input file.")
            return
            
        # Determine output path
        if not self.output_file:
            input_name = os.path.splitext(os.path.basename(self.input_file))[0]
            self.output_file = f"{input_name}_visa.jpg"
            
        # Update settings
        self.update_settings()
        
        # Start processing
        self.show_progress(True)
        self.process_btn.setEnabled(False)
        
        # Create processing thread
        self.processing_thread = ProcessingThread(
            self.processor, self.input_file, "visa"
        )
        
        # Connect signals
        self.processing_thread.progress_updated.connect(self.update_progress)
        self.processing_thread.status_updated.connect(self.update_status)
        self.processing_thread.finished_processing.connect(self.on_processing_finished)
        self.processing_thread.error_occurred.connect(self.on_processing_error)
        
        # Start processing
        self.processing_thread.start()
        
    def show_progress(self, show: bool):
        """Show or hide progress indicators."""
        self.progress_group.setVisible(show)
        if not show:
            self.progress_bar.setValue(0)
            self.progress_label.setText("")
            
    def update_progress(self, value: int):
        """Update progress bar."""
        self.progress_bar.setValue(value)
        
    def update_status(self, message: str):
        """Update status message."""
        self.progress_label.setText(message)
        
    def on_processing_finished(self, results):
        """Handle processing completion."""
        self.show_progress(False)
        self.process_btn.setEnabled(True)
        
        if results.get('success', False):
            # Display success message
            text = f"✅ Visa photo created successfully!\\n\\n"
            text += f"📁 Output file: {results['output_path']}\\n"
            
            dims = results['dimensions']
            text += f"📏 Dimensions: {dims['width_mm']}×{dims['height_mm']}mm\\n"
            text += f"🔍 Resolution: {dims['dpi']} DPI\\n"
            text += f"💾 Size: {dims['width_px']}×{dims['height_px']} pixels\\n\\n"
            
            validation = results['validation']
            if validation['valid']:
                text += "✅ Photo meets visa requirements"
            else:
                text += "⚠️ Validation issues:\\n"
                for issue in validation['issues']:
                    text += f"  • {issue}\\n"
                    
            self.results_text.setText(text)
            
            # Show success dialog
            QMessageBox.information(
                self,
                "Success",
                f"Visa photo created successfully!\\n\\nSaved to: {results['output_path']}"
            )
        else:
            self.results_text.setText(f"❌ Processing failed: {results.get('error', 'Unknown error')}")
            
    def on_processing_error(self, error_message: str):
        """Handle processing error."""
        self.show_progress(False)
        self.process_btn.setEnabled(True)
        
        self.results_text.setText(f"❌ Error: {error_message}")
        QMessageBox.critical(self, "Processing Error", f"Visa photo creation failed:\\n\\n{error_message}")