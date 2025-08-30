"""Settings dialog for Photo AI configuration."""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QTabWidget,
    QLabel, QPushButton, QSpinBox, QDoubleSpinBox, QLineEdit,
    QCheckBox, QComboBox, QGroupBox, QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from ...core.config import Config


class SettingsDialog(QDialog):
    """Dialog for configuring Photo AI settings."""
    
    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self.config = config
        self.setup_ui()
        self.load_settings()
        self.setup_connections()
        
    def setup_ui(self):
        """Setup the settings dialog UI."""
        self.setWindowTitle("Photo AI Settings")
        self.setModal(True)
        self.resize(600, 500)
        
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Settings")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("margin: 10px; color: #2196F3;")
        layout.addWidget(title)
        
        # Tab widget for different setting categories
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.create_general_tab()
        self.create_processing_tab()
        self.create_visa_tab()
        self.create_directories_tab()
        
        # Button layout
        button_layout = QHBoxLayout()
        
        self.reset_btn = QPushButton("Reset to Defaults")
        button_layout.addWidget(self.reset_btn)
        
        button_layout.addStretch()
        
        self.cancel_btn = QPushButton("Cancel")
        button_layout.addWidget(self.cancel_btn)
        
        self.apply_btn = QPushButton("Apply")
        button_layout.addWidget(self.apply_btn)
        
        self.ok_btn = QPushButton("OK")
        self.ok_btn.setDefault(True)
        button_layout.addWidget(self.ok_btn)
        
        layout.addLayout(button_layout)
        
    def create_general_tab(self):
        """Create the general settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Model settings
        model_group = QGroupBox("Model Settings")
        model_layout = QGridLayout(model_group)
        
        model_layout.addWidget(QLabel("Device:"), 0, 0)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu", "cuda"])
        model_layout.addWidget(self.device_combo, 0, 1)
        
        model_layout.addWidget(QLabel("Sharpness Model:"), 1, 0)
        self.sharpness_model_edit = QLineEdit()
        model_layout.addWidget(self.sharpness_model_edit, 1, 1)
        
        model_layout.addWidget(QLabel("Feature Model:"), 2, 0)
        self.feature_model_edit = QLineEdit()
        model_layout.addWidget(self.feature_model_edit, 2, 1)
        
        layout.addWidget(model_group)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "General")
        
    def create_processing_tab(self):
        """Create the processing settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Processing parameters
        processing_group = QGroupBox("Processing Parameters")
        processing_layout = QGridLayout(processing_group)
        
        processing_layout.addWidget(QLabel("Time Threshold (seconds):"), 0, 0)
        self.time_threshold_spin = QSpinBox()
        self.time_threshold_spin.setRange(30, 3600)
        processing_layout.addWidget(self.time_threshold_spin, 0, 1)
        
        processing_layout.addWidget(QLabel("Cluster EPS:"), 1, 0)
        self.cluster_eps_spin = QDoubleSpinBox()
        self.cluster_eps_spin.setRange(0.1, 2.0)
        self.cluster_eps_spin.setSingleStep(0.1)
        self.cluster_eps_spin.setDecimals(2)
        processing_layout.addWidget(self.cluster_eps_spin, 1, 1)
        
        processing_layout.addWidget(QLabel("Min Photos to Cluster:"), 2, 0)
        self.min_photos_spin = QSpinBox()
        self.min_photos_spin.setRange(2, 20)
        processing_layout.addWidget(self.min_photos_spin, 2, 1)
        
        processing_layout.addWidget(QLabel("Sharpness Threshold:"), 3, 0)
        self.sharpness_threshold_spin = QDoubleSpinBox()
        self.sharpness_threshold_spin.setRange(0.1, 1.0)
        self.sharpness_threshold_spin.setSingleStep(0.05)
        self.sharpness_threshold_spin.setDecimals(2)
        processing_layout.addWidget(self.sharpness_threshold_spin, 3, 1)
        
        processing_layout.addWidget(QLabel("Best Photos Count:"), 4, 0)
        self.num_best_photos_spin = QSpinBox()
        self.num_best_photos_spin.setRange(1, 10)
        processing_layout.addWidget(self.num_best_photos_spin, 4, 1)
        
        layout.addWidget(processing_group)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "Processing")
        
    def create_visa_tab(self):
        """Create the visa photo settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Visa photo parameters
        visa_group = QGroupBox("Visa Photo Parameters")
        visa_layout = QGridLayout(visa_group)
        
        visa_layout.addWidget(QLabel("DPI:"), 0, 0)
        self.visa_dpi_spin = QSpinBox()
        self.visa_dpi_spin.setRange(150, 600)
        visa_layout.addWidget(self.visa_dpi_spin, 0, 1)
        
        visa_layout.addWidget(QLabel("Width (mm):"), 1, 0)
        self.visa_width_spin = QSpinBox()
        self.visa_width_spin.setRange(20, 100)
        visa_layout.addWidget(self.visa_width_spin, 1, 1)
        
        visa_layout.addWidget(QLabel("Height (mm):"), 2, 0)
        self.visa_height_spin = QSpinBox()
        self.visa_height_spin.setRange(20, 150)
        visa_layout.addWidget(self.visa_height_spin, 2, 1)
        
        visa_layout.addWidget(QLabel("Face Height Ratio:"), 3, 0)
        self.face_height_ratio_spin = QDoubleSpinBox()
        self.face_height_ratio_spin.setRange(0.2, 0.8)
        self.face_height_ratio_spin.setSingleStep(0.05)
        self.face_height_ratio_spin.setDecimals(2)
        visa_layout.addWidget(self.face_height_ratio_spin, 3, 1)
        
        visa_layout.addWidget(QLabel("Face Top Margin Ratio:"), 4, 0)
        self.face_top_margin_spin = QDoubleSpinBox()
        self.face_top_margin_spin.setRange(0.05, 0.3)
        self.face_top_margin_spin.setSingleStep(0.01)
        self.face_top_margin_spin.setDecimals(2)
        visa_layout.addWidget(self.face_top_margin_spin, 4, 1)
        
        layout.addWidget(visa_group)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "Visa Photos")
        
    def create_directories_tab(self):
        """Create the directories settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Directory settings
        dir_group = QGroupBox("Directory Settings")
        dir_layout = QGridLayout(dir_group)
        
        # Input directory
        dir_layout.addWidget(QLabel("Input Directory:"), 0, 0)
        self.input_dir_edit = QLineEdit()
        dir_layout.addWidget(self.input_dir_edit, 0, 1)
        self.input_dir_btn = QPushButton("Browse...")
        dir_layout.addWidget(self.input_dir_btn, 0, 2)
        
        # Good directory
        dir_layout.addWidget(QLabel("Good Photos Directory:"), 1, 0)
        self.good_dir_edit = QLineEdit()
        dir_layout.addWidget(self.good_dir_edit, 1, 1)
        self.good_dir_btn = QPushButton("Browse...")
        dir_layout.addWidget(self.good_dir_btn, 1, 2)
        
        # Bad directory
        dir_layout.addWidget(QLabel("Bad Photos Directory:"), 2, 0)
        self.bad_dir_edit = QLineEdit()
        dir_layout.addWidget(self.bad_dir_edit, 2, 1)
        self.bad_dir_btn = QPushButton("Browse...")
        dir_layout.addWidget(self.bad_dir_btn, 2, 2)
        
        # Output directory
        dir_layout.addWidget(QLabel("Output Directory:"), 3, 0)
        self.output_dir_edit = QLineEdit()
        dir_layout.addWidget(self.output_dir_edit, 3, 1)
        self.output_dir_btn = QPushButton("Browse...")
        dir_layout.addWidget(self.output_dir_btn, 3, 2)
        
        layout.addWidget(dir_group)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "Directories")
        
    def setup_connections(self):
        """Setup signal connections."""
        # Directory browse buttons
        self.input_dir_btn.clicked.connect(lambda: self.browse_directory(self.input_dir_edit))
        self.good_dir_btn.clicked.connect(lambda: self.browse_directory(self.good_dir_edit))
        self.bad_dir_btn.clicked.connect(lambda: self.browse_directory(self.bad_dir_edit))
        self.output_dir_btn.clicked.connect(lambda: self.browse_directory(self.output_dir_edit))
        
        # Main buttons
        self.reset_btn.clicked.connect(self.reset_to_defaults)
        self.cancel_btn.clicked.connect(self.reject)
        self.apply_btn.clicked.connect(self.apply_settings)
        self.ok_btn.clicked.connect(self.accept_settings)
        
    def load_settings(self):
        """Load current settings into the UI."""
        # General settings
        self.device_combo.setCurrentText(self.config.models.device)
        self.sharpness_model_edit.setText(self.config.models.sharpness_model)
        self.feature_model_edit.setText(self.config.models.feature_model)
        
        # Processing settings
        self.time_threshold_spin.setValue(self.config.processing.time_threshold)
        self.cluster_eps_spin.setValue(self.config.processing.cluster_eps)
        self.min_photos_spin.setValue(self.config.processing.min_photos_to_cluster)
        self.sharpness_threshold_spin.setValue(self.config.processing.sharpness_threshold)
        self.num_best_photos_spin.setValue(self.config.processing.num_best_photos)
        
        # Visa settings
        self.visa_dpi_spin.setValue(self.config.visa.dpi)
        self.visa_width_spin.setValue(self.config.visa.photo_width_mm)
        self.visa_height_spin.setValue(self.config.visa.photo_height_mm)
        self.face_height_ratio_spin.setValue(self.config.visa.face_height_ratio)
        self.face_top_margin_spin.setValue(self.config.visa.face_top_margin_ratio)
        
        # Directory settings
        self.input_dir_edit.setText(self.config.input_dir)
        self.good_dir_edit.setText(self.config.good_dir)
        self.bad_dir_edit.setText(self.config.bad_dir)
        self.output_dir_edit.setText(self.config.output_dir)
        
    def apply_settings(self):
        """Apply settings to the config object."""
        # General settings
        self.config.models.device = self.device_combo.currentText()
        self.config.models.sharpness_model = self.sharpness_model_edit.text()
        self.config.models.feature_model = self.feature_model_edit.text()
        
        # Processing settings
        self.config.processing.time_threshold = self.time_threshold_spin.value()
        self.config.processing.cluster_eps = self.cluster_eps_spin.value()
        self.config.processing.min_photos_to_cluster = self.min_photos_spin.value()
        self.config.processing.sharpness_threshold = self.sharpness_threshold_spin.value()
        self.config.processing.num_best_photos = self.num_best_photos_spin.value()
        
        # Visa settings
        self.config.visa.dpi = self.visa_dpi_spin.value()
        self.config.visa.photo_width_mm = self.visa_width_spin.value()
        self.config.visa.photo_height_mm = self.visa_height_spin.value()
        self.config.visa.face_height_ratio = self.face_height_ratio_spin.value()
        self.config.visa.face_top_margin_ratio = self.face_top_margin_spin.value()
        
        # Directory settings
        self.config.input_dir = self.input_dir_edit.text()
        self.config.good_dir = self.good_dir_edit.text()
        self.config.bad_dir = self.bad_dir_edit.text()
        self.config.output_dir = self.output_dir_edit.text()
        
    def accept_settings(self):
        """Apply settings and close dialog."""
        self.apply_settings()
        self.accept()
        
    def reset_to_defaults(self):
        """Reset all settings to default values."""
        reply = QMessageBox.question(
            self,
            "Reset Settings",
            "Are you sure you want to reset all settings to default values?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Create a new default config
            default_config = Config()
            
            # Copy default values to current config
            self.config.models = default_config.models
            self.config.processing = default_config.processing
            self.config.visa = default_config.visa
            
            # Reload the UI
            self.load_settings()
            
    def browse_directory(self, line_edit: QLineEdit):
        """Browse for a directory."""
        current_dir = line_edit.text() or ""
        
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Directory",
            current_dir,
            QFileDialog.Option.ShowDirsOnly
        )
        
        if directory:
            line_edit.setText(directory)