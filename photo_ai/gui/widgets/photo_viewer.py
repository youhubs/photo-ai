"""Photo viewer widget for displaying and browsing photos."""

import os
from typing import List, Optional
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QFrame,
    QSplitter,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QFont, QPainter, QPen, QIcon, QKeySequence, QShortcut


class PhotoThumbnailLoader(QThread):
    """Thread for loading photo thumbnails in the background."""

    thumbnail_loaded = pyqtSignal(str, QPixmap)

    def __init__(self, photo_paths: List[str], fast_mode: bool = False):
        super().__init__()
        self.photo_paths = photo_paths
        self.thumbnail_size = QSize(120, 120) if fast_mode else QSize(150, 150)
        self.fast_mode = fast_mode

    def run(self):
        """Load thumbnails for all photos."""
        for photo_path in self.photo_paths:
            try:
                pixmap = QPixmap(photo_path)
                if not pixmap.isNull():
                    # Use faster scaling for fast mode
                    transform_mode = (
                        Qt.TransformationMode.FastTransformation
                        if self.fast_mode
                        else Qt.TransformationMode.SmoothTransformation
                    )

                    # Scale to thumbnail size while maintaining aspect ratio
                    thumbnail = pixmap.scaled(
                        self.thumbnail_size,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        transform_mode,
                    )
                    self.thumbnail_loaded.emit(photo_path, thumbnail)
            except Exception as e:
                print(f"Failed to load thumbnail for {photo_path}: {e}")


class PhotoViewer(QWidget):
    """Widget for viewing and browsing photos."""

    def __init__(self):
        super().__init__()
        self.photo_paths: List[str] = []
        self.current_index = 0
        self.thumbnail_loader: Optional[PhotoThumbnailLoader] = None
        self.total_photo_count: Optional[int] = None  # For fast loading mode
        self.full_photo_paths: Optional[List[str]] = None  # All photos when loaded
        self.current_folder: Optional[str] = None  # Store current folder path

        self.setup_ui()

    def setup_ui(self):
        """Setup the photo viewer interface."""
        layout = QVBoxLayout(self)

        # Create splitter for main view and thumbnails
        splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(splitter)

        # Main photo display area
        self.main_photo_area = self.create_main_photo_area()
        splitter.addWidget(self.main_photo_area)

        # Thumbnail area
        self.thumbnail_area = self.create_thumbnail_area()
        splitter.addWidget(self.thumbnail_area)

        # Set splitter proportions
        splitter.setSizes([600, 200])

        # Navigation controls
        nav_layout = QHBoxLayout()

        self.prev_btn = QPushButton("◀ Previous")
        self.prev_btn.clicked.connect(self.show_previous)
        self.prev_btn.setEnabled(False)
        nav_layout.addWidget(self.prev_btn)

        self.photo_counter = QLabel("No photos loaded")
        self.photo_counter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav_layout.addWidget(self.photo_counter)

        # Delete button for removing bad photos
        self.delete_btn = QPushButton("🗑️ Delete Photo")
        self.delete_btn.clicked.connect(self.delete_current_photo)
        self.delete_btn.setEnabled(False)
        self.delete_btn.setToolTip(
            "Delete the current photo from disk (permanent)\nKeyboard shortcut: Delete key"
        )
        self.delete_btn.setStyleSheet(
            "QPushButton { background-color: #ff4444; color: white; font-weight: bold; }"
        )
        nav_layout.addWidget(self.delete_btn)

        # Load All Photos button (initially hidden)
        self.load_all_btn = QPushButton("📁 Load All Photos")
        self.load_all_btn.clicked.connect(self.load_all_photos)
        self.load_all_btn.setVisible(False)
        self.load_all_btn.setToolTip("Load all photos in the folder for browsing")
        self.load_all_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }"
        )
        nav_layout.addWidget(self.load_all_btn)

        self.next_btn = QPushButton("Next ▶")
        self.next_btn.clicked.connect(self.show_next)
        self.next_btn.setEnabled(False)
        nav_layout.addWidget(self.next_btn)

        layout.addLayout(nav_layout)

        # Add keyboard shortcut for delete
        self.delete_shortcut = QShortcut(QKeySequence.StandardKey.Delete, self)
        self.delete_shortcut.activated.connect(self.delete_current_photo)

    def create_main_photo_area(self) -> QWidget:
        """Create the main photo display area."""
        area = QFrame()
        area.setFrameStyle(QFrame.Shape.StyledPanel)
        area.setMinimumHeight(400)

        layout = QVBoxLayout(area)

        # Photo info label
        self.photo_info_label = QLabel()
        self.photo_info_label.setFont(QFont("Arial", 10))
        self.photo_info_label.setStyleSheet("color: #888888; padding: 5px;")
        layout.addWidget(self.photo_info_label)

        # Scroll area for large photos
        scroll_area = QScrollArea()
        scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scroll_area.setStyleSheet("QScrollArea { border: none; }")
        layout.addWidget(scroll_area)

        # Photo display label
        self.photo_label = QLabel()
        self.photo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.photo_label.setStyleSheet(
            """
            QLabel {
                background-color: #2a2a2a;
                border: 2px dashed #555555;
                border-radius: 10px;
                color: #888888;
                font-size: 16px;
            }
        """
        )
        self.photo_label.setText("No photo selected\\n\\nSelect a folder or files to begin")
        self.photo_label.setMinimumSize(400, 300)

        scroll_area.setWidget(self.photo_label)
        scroll_area.setWidgetResizable(True)

        return area

    def create_thumbnail_area(self) -> QWidget:
        """Create the thumbnail browser area."""
        area = QFrame()
        area.setFrameStyle(QFrame.Shape.StyledPanel)
        area.setMaximumHeight(200)

        layout = QVBoxLayout(area)

        # Thumbnail list
        self.thumbnail_list = QListWidget()
        self.thumbnail_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.thumbnail_list.setIconSize(QSize(120, 120))
        self.thumbnail_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.thumbnail_list.setMovement(QListWidget.Movement.Static)
        self.thumbnail_list.setSpacing(5)
        self.thumbnail_list.itemClicked.connect(self.on_thumbnail_clicked)

        layout.addWidget(self.thumbnail_list)

        return area

    def delete_current_photo(self):
        """Delete the currently displayed photo from disk."""
        if not self.photo_paths or self.current_index >= len(self.photo_paths):
            return

        current_photo_path = self.photo_paths[self.current_index]
        filename = os.path.basename(current_photo_path)

        # Confirmation dialog
        reply = QMessageBox.question(
            self,
            "Delete Photo",
            f"Are you sure you want to permanently delete:\n\n{filename}\n\nThis action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                # Delete the file from disk
                os.remove(current_photo_path)
                print(f"Deleted photo: {current_photo_path}")

                # Remove from our photo lists
                self.photo_paths.pop(self.current_index)
                if self.full_photo_paths and current_photo_path in self.full_photo_paths:
                    self.full_photo_paths.remove(current_photo_path)

                # Update total count
                if self.total_photo_count:
                    self.total_photo_count -= 1

                # Remove thumbnail from list
                self.remove_thumbnail_at_index(self.current_index)

                # Adjust current index if needed
                if self.current_index >= len(self.photo_paths) and self.photo_paths:
                    self.current_index = len(self.photo_paths) - 1
                elif not self.photo_paths:
                    self.current_index = 0

                # Update UI
                if self.photo_paths:
                    self.show_current_photo()
                    self.update_photo_counter()
                    self.update_navigation_buttons()
                else:
                    self.clear_photos()

                # Show success message briefly
                QMessageBox.information(
                    self,
                    "Photo Deleted",
                    f"Successfully deleted {filename}",
                    QMessageBox.StandardButton.Ok,
                )

            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Delete Failed",
                    f"Failed to delete {filename}:\n\n{str(e)}",
                    QMessageBox.StandardButton.Ok,
                )

    def remove_thumbnail_at_index(self, index: int):
        """Remove thumbnail at specific index from the thumbnail list."""
        if index < self.thumbnail_list.count():
            item = self.thumbnail_list.takeItem(index)
            if item:
                del item

    def load_all_photos(self):
        """Load all photos from the current folder."""
        if not self.current_folder:
            print("No current folder to load from")
            return

        try:
            # Import here to avoid circular imports
            from ...utils.image_utils import get_image_paths

            # Update button to show loading state
            self.load_all_btn.setText("🔄 Loading All Photos...")
            self.load_all_btn.setEnabled(False)

            # Get all image paths from folder
            all_paths = get_image_paths(self.current_folder)

            if all_paths:
                self.full_photo_paths = all_paths
                self.photo_paths = all_paths
                self.total_photo_count = len(all_paths)

                # Update UI
                self.update_photo_counter()
                self.update_navigation_buttons()

                # Load all thumbnails
                self.load_thumbnails()

                # Hide the Load All button since all photos are now loaded
                self.load_all_btn.setVisible(False)

                print(f"Loaded {len(all_paths)} photos for browsing")
            else:
                print("No photos found in folder")
                self.load_all_btn.setText("❌ No Photos Found")

        except Exception as e:
            print(f"Error loading all photos: {e}")
            self.load_all_btn.setText("❌ Error Loading")
            self.load_all_btn.setEnabled(True)

    def load_photos(self, photo_paths: List[str]):
        """Load photos into the viewer."""
        self.photo_paths = photo_paths
        self.current_index = 0
        self.full_photo_paths = photo_paths  # All photos are already loaded
        self.total_photo_count = len(photo_paths)
        self.current_folder = None  # Clear folder since we have specific files

        if not photo_paths:
            self.clear_photos()
            return

        # Hide Load All button since all photos are already loaded
        self.load_all_btn.setVisible(False)

        # Update counter
        self.update_photo_counter()

        # Enable/disable navigation
        self.update_navigation_buttons()

        # Show first photo
        self.show_current_photo()

        # Load thumbnails in background
        self.load_thumbnails()

    def load_photos_fast(
        self, preview_paths: List[str], total_count: int, folder_path: Optional[str] = None
    ):
        """Load photos with fast preview (only first few photos loaded initially)."""
        self.photo_paths = preview_paths  # Start with preview paths
        self.total_photo_count = total_count  # Store total count
        self.current_index = 0
        self.full_photo_paths = None  # Will be set when all photos are loaded
        self.current_folder = folder_path  # Store folder for loading all photos later

        if not preview_paths:
            self.clear_photos()
            return

        # Show Load All Photos button if there are more photos than preview
        if total_count > len(preview_paths):
            self.load_all_btn.setVisible(True)
            self.load_all_btn.setText(f"📁 Load All {total_count} Photos")
        else:
            self.load_all_btn.setVisible(False)

        # Update counter with total count
        self.update_photo_counter_with_total(total_count)

        # Enable/disable navigation
        self.update_navigation_buttons()

        # Show first photo
        self.show_current_photo()

        # Load thumbnails for preview photos only
        self.load_thumbnails_fast(preview_paths)

    def clear_photos(self):
        """Clear all photos from viewer."""
        self.photo_paths.clear()
        self.current_index = 0
        self.thumbnail_list.clear()
        self.total_photo_count = None
        self.full_photo_paths = None
        self.current_folder = None

        self.photo_label.setText("No photo selected\\n\\nSelect a folder or files to begin")
        self.photo_counter.setText("No photos loaded")
        self.photo_info_label.setText("")
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.delete_btn.setEnabled(False)
        self.load_all_btn.setVisible(False)

    def load_thumbnails(self):
        """Load thumbnails in a background thread."""
        if self.thumbnail_loader and self.thumbnail_loader.isRunning():
            self.thumbnail_loader.terminate()

        self.thumbnail_list.clear()

        # Add placeholder items first
        for i, path in enumerate(self.photo_paths):
            item = QListWidgetItem(os.path.basename(path))
            item.setData(Qt.ItemDataRole.UserRole, path)
            self.thumbnail_list.addItem(item)

        # Start loading thumbnails
        self.thumbnail_loader = PhotoThumbnailLoader(self.photo_paths)
        self.thumbnail_loader.thumbnail_loaded.connect(self.on_thumbnail_loaded)
        self.thumbnail_loader.start()

    def load_thumbnails_fast(self, preview_paths: List[str]):
        """Load thumbnails for preview photos only (faster loading)."""
        if self.thumbnail_loader and self.thumbnail_loader.isRunning():
            self.thumbnail_loader.terminate()

        self.thumbnail_list.clear()

        # Add placeholder items for preview photos only
        for i, path in enumerate(preview_paths):
            item = QListWidgetItem(os.path.basename(path))
            item.setData(Qt.ItemDataRole.UserRole, path)
            self.thumbnail_list.addItem(item)

        # Start loading thumbnails for preview photos (with fast mode)
        self.thumbnail_loader = PhotoThumbnailLoader(preview_paths, fast_mode=True)
        self.thumbnail_loader.thumbnail_loaded.connect(self.on_thumbnail_loaded)
        self.thumbnail_loader.start()

    def on_thumbnail_loaded(self, photo_path: str, pixmap: QPixmap):
        """Handle loaded thumbnail."""
        # Find the corresponding list item
        for i in range(self.thumbnail_list.count()):
            item = self.thumbnail_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == photo_path:
                item.setIcon(QIcon(pixmap))
                break

    def on_thumbnail_clicked(self, item: QListWidgetItem):
        """Handle thumbnail click."""
        photo_path = item.data(Qt.ItemDataRole.UserRole)
        try:
            self.current_index = self.photo_paths.index(photo_path)
            self.show_current_photo()
            self.update_photo_counter()
            self.update_navigation_buttons()
        except ValueError:
            pass

    def show_current_photo(self):
        """Show the current photo in the main display."""
        if not self.photo_paths or self.current_index >= len(self.photo_paths):
            return

        photo_path = self.photo_paths[self.current_index]

        try:
            # Load and display photo
            pixmap = QPixmap(photo_path)
            if pixmap.isNull():
                self.photo_label.setText(f"Failed to load image:\\n{os.path.basename(photo_path)}")
                return

            # Scale photo to fit display
            max_size = QSize(800, 600)
            scaled_pixmap = pixmap.scaled(
                max_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

            self.photo_label.setPixmap(scaled_pixmap)

            # Update photo info
            self.update_photo_info(photo_path, pixmap.size())

        except Exception as e:
            self.photo_label.setText(f"Error loading image:\\n{str(e)}")

    def update_photo_info(self, photo_path: str, original_size: QSize):
        """Update photo information display."""
        filename = os.path.basename(photo_path)
        file_size = self.get_file_size(photo_path)

        info = (
            f"📁 {filename} | 📏 {original_size.width()}×{original_size.height()} | 💾 {file_size}"
        )
        self.photo_info_label.setText(info)

    def get_file_size(self, file_path: str) -> str:
        """Get formatted file size."""
        try:
            size = os.path.getsize(file_path)
            for unit in ["B", "KB", "MB", "GB"]:
                if size < 1024.0:
                    return f"{size:.1f} {unit}"
                size /= 1024.0
            return f"{size:.1f} TB"
        except:
            return "Unknown"

    def show_previous(self):
        """Show previous photo."""
        if self.current_index > 0:
            self.current_index -= 1
            self.show_current_photo()
            self.update_photo_counter()
            self.update_navigation_buttons()

    def show_next(self):
        """Show next photo."""
        if self.current_index < len(self.photo_paths) - 1:
            self.current_index += 1
            self.show_current_photo()
            self.update_photo_counter()
            self.update_navigation_buttons()

    def update_photo_counter(self):
        """Update the photo counter display."""
        if self.photo_paths:
            total = len(self.photo_paths)
            current = self.current_index + 1
            self.photo_counter.setText(f"Photo {current} of {total}")
        else:
            self.photo_counter.setText("No photos loaded")

    def update_photo_counter_with_total(self, total_count: int):
        """Update the photo counter with total count (for fast loading mode)."""
        if self.photo_paths:
            current = self.current_index + 1
            preview_count = len(self.photo_paths)
            if total_count > preview_count:
                self.photo_counter.setText(
                    f"Photo {current} of {preview_count} (preview of {total_count} total)"
                )
            else:
                self.photo_counter.setText(f"Photo {current} of {total_count}")
        else:
            self.photo_counter.setText("No photos loaded")

    def update_navigation_buttons(self):
        """Update navigation button states."""
        has_photos = bool(self.photo_paths)
        self.prev_btn.setEnabled(has_photos and self.current_index > 0)
        self.next_btn.setEnabled(has_photos and self.current_index < len(self.photo_paths) - 1)
        self.delete_btn.setEnabled(has_photos)  # Enable delete button when photos are loaded
