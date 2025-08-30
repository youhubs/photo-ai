"""Main PyQt application class."""

import sys
from typing import Optional
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QPalette, QColor

from .main_window import PhotoAIMainWindow


class PhotoAIApp(QApplication):
    """Main Photo AI desktop application."""

    def __init__(self, argv):
        super().__init__(argv)

        self.setApplicationName("Photo AI")
        self.setApplicationVersion("1.0.0")
        self.setOrganizationName("Photo AI Team")

        # Set application properties
        self.setQuitOnLastWindowClosed(True)

        # Apply modern dark theme
        self.apply_dark_theme()

        # Create main window
        self.main_window = PhotoAIMainWindow()

    def apply_dark_theme(self):
        """Apply a modern dark theme to the application."""
        self.setStyle("Fusion")

        palette = QPalette()

        # Window colors
        palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))

        # Base colors (for input fields)
        palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))

        # Text colors
        palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))

        # Button colors
        palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))

        # Highlight colors
        palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))

        # Disabled colors
        palette.setColor(
            QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(127, 127, 127)
        )
        palette.setColor(
            QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(127, 127, 127)
        )
        palette.setColor(
            QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(127, 127, 127)
        )

        self.setPalette(palette)

        # Set stylesheet for additional styling
        self.setStyleSheet(
            """
            QToolTip {
                color: #ffffff;
                background-color: #2a2a2a;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 3px;
            }
            
            QPushButton {
                background-color: #404040;
                border: 1px solid #555555;
                border-radius: 5px;
                padding: 8px 16px;
                font-weight: bold;
            }
            
            QPushButton:hover {
                background-color: #505050;
                border-color: #777777;
            }
            
            QPushButton:pressed {
                background-color: #353535;
            }
            
            QPushButton:disabled {
                background-color: #2a2a2a;
                color: #666666;
            }
            
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 5px;
                text-align: center;
            }
            
            QProgressBar::chunk {
                background-color: #42a5f5;
                border-radius: 4px;
            }
            
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 5px;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """
        )

    def run(self):
        """Start the application."""
        self.main_window.show()
        return self.exec()


def main():
    """Entry point for the GUI application."""
    app = PhotoAIApp(sys.argv)
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
