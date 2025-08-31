# gui/widget/logger_widget.py
from PyQt6.QtWidgets import QTextEdit
from PyQt6.QtGui import QColor, QTextCursor


class LoggerWidget(QTextEdit):
    """
    A QTextEdit-based logger widget for PyQt6 that supports
    colored messages and auto-scroll.
    """

    LEVEL_COLORS = {
        "info": "#3498db",      # blue
        "success": "#2ecc71",   # green
        "warning": "#f1c40f",   # yellow
        "error": "#e74c3c",     # red
    }

    def __init__(self, parent=None, dark_mode=True):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.setCursorWidth(2)

        # Apply default dark or light style
        if dark_mode:
            self.setStyleSheet("""
                background-color: #2a2a2a;
                color: #ffffff;
                font-family: monospace;
                font-size: 12pt;
            """)
        else:
            self.setStyleSheet("""
                background-color: #ffffff;
                color: #000000;
                font-family: monospace;
                font-size: 12pt;
            """)

    def log(self, message: str, level: str = "info"):
        """Append a message with optional level color."""
        color = self.LEVEL_COLORS.get(level, "#ffffff")
        # Escape HTML
        message_html = message.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.append(f'<span style="color:{color}">{message_html}</span>')
        self._scroll_to_bottom()

    def log_info(self, message: str):
        self.log(message, level="info")

    def log_success(self, message: str):
        self.log(message, level="success")

    def log_warning(self, message: str):
        self.log(message, level="warning")

    def log_error(self, message: str):
        self.log(message, level="error")

    def _scroll_to_bottom(self):
        """Auto-scroll to the latest message."""
        self.moveCursor(QTextCursor.MoveOperation.End)
        self.ensureCursorVisible()
