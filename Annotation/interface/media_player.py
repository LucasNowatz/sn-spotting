# Adapted from https://codeloop.org/python-how-to-create-media-player-in-pyqt5/
import os
from PyQt5.QtWidgets import QWidget, QPushButton, QStyle, QSlider, QHBoxLayout, QVBoxLayout, QFileDialog
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import Qt, QUrl


class MediaPlayer(QWidget):

	def __init__(self, main_window):

		super().__init__()

		self.main_window = main_window
		self.path_label = None

		self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
		self.video_widget = QVideoWidget()

		self.open_file_button = QPushButton('Open video')
		self.open_file_button.clicked.connect(self.open_file)

		self.play_button = QPushButton()
		self.play_button.setEnabled(False)
		self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
		self.play_button.clicked.connect(self.play_video)

		self.slider = QSlider(Qt.Horizontal)
		self.slider.setRange(0, 0)
		self.slider.sliderMoved.connect(self.set_position)

		hboxLayout = QHBoxLayout()
		hboxLayout.setContentsMargins(0, 0, 0, 0)
		hboxLayout.addWidget(self.open_file_button)
		hboxLayout.addWidget(self.play_button)
		hboxLayout.addWidget(self.slider)

		self.layout = QVBoxLayout()
		self.layout.addWidget(self.video_widget)
		self.layout.addLayout(hboxLayout)

		self.media_player.setVideoOutput(self.video_widget)
		self.media_player.stateChanged.connect(self.mediastate_changed)
		self.media_player.positionChanged.connect(self.position_changed)
		self.media_player.durationChanged.connect(self.duration_changed)

	def load_video(self, filename: str) -> None:
		from utils.dataset_catalog import infer_half, resolve_label_path

		filename = os.path.abspath(filename)
		if not os.path.isfile(filename):
			print(f"Video not found: {filename}")
			return

		self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(filename)))
		self.play_button.setEnabled(True)
		self.main_window.half = infer_half(filename)

		self.path_label = str(resolve_label_path(filename))
		self.main_window.list_manager.create_list_from_json(self.path_label, self.main_window.half)
		self.main_window.list_display.display_list(self.main_window.list_manager.create_text_list())
		self.main_window.setWindowTitle(f"Event Annotator — {os.path.basename(filename)}")

	def open_file(self):
		filename, _ = QFileDialog.getOpenFileName(self, "Open Video")
		if filename:
			self.load_video(filename)

	def get_last_label_file(self):
		path_label = self.path_label
		folder_label = os.path.dirname(path_label)
		for name in ("ground_truth.json", "Labels-v2.json", "Labels.json"):
			candidate = os.path.join(folder_label, name)
			if os.path.isfile(candidate):
				return candidate
		return path_label

	def play_video(self):
		if self.media_player.state() == QMediaPlayer.PlayingState:
			self.media_player.pause()
		else:
			self.media_player.play()

	def mediastate_changed(self, state):
		if self.media_player.state() == QMediaPlayer.PlayingState:
			self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
		else:
			self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

	def position_changed(self, position):
		self.slider.setValue(position)

	def duration_changed(self, duration):
		self.slider.setRange(0, duration)

	def set_position(self, position):
		self.media_player.setPosition(position)

	def handle_errors(self):
		self.play_button.setEnabled(False)
		print("Error: " + self.media_player.errorString())
