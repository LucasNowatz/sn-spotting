from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QListWidget, QVBoxLayout, QWidget

from utils.dataset_catalog import (
    discover_videos,
    display_name,
    resolve_dataset_root,
)


class VideoBrowser(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.dataset_root = None
        self.video_paths: list[str] = []

        self.setMaximumWidth(420)

        self.title = QLabel("Dataset videos")
        self.title.setStyleSheet("color: white; font-weight: bold;")

        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet(
            "QListWidget { background: #1e1e1e; color: #f0f0f0; border: 1px solid #555; }"
            "QListWidget::item:selected { background: #007acc; color: white; }"
        )
        self.list_widget.itemClicked.connect(self._on_item_activated)
        self.list_widget.itemDoubleClicked.connect(self._on_item_activated)
        self.list_widget.itemActivated.connect(self._on_item_activated)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.title)
        layout.addWidget(self.list_widget)
        self.setLayout(layout)

    def refresh(self, *, auto_load_first: bool = True) -> int:
        self.list_widget.clear()
        self.video_paths.clear()
        try:
            self.dataset_root = resolve_dataset_root()
        except FileNotFoundError as exc:
            self.title.setText(f"Dataset videos — {exc}")
            return 0

        videos = discover_videos(self.dataset_root)
        self.title.setText(f"Dataset videos ({len(videos)}) — click to open")
        for video in videos:
            self.video_paths.append(str(video))
            self.list_widget.addItem(display_name(video, self.dataset_root))

        if auto_load_first and self.video_paths:
            self.list_widget.setCurrentRow(0)
            self.main_window.media_player.load_video(self.video_paths[0])
        return len(self.video_paths)

    def _on_item_activated(self, item):
        row = self.list_widget.row(item)
        if 0 <= row < len(self.video_paths):
            self.main_window.media_player.load_video(self.video_paths[row])
