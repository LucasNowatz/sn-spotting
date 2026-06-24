from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import Qt
from PyQt5.QtMultimedia import QMediaPlayer

from interface.media_player import MediaPlayer
from interface.list_display import ListDisplay
from interface.video_browser import VideoBrowser
from interface.event_selection import EventSelectionWindow
from utils.list_management import ListManager
from utils.event_class import Event, ms_to_time

class MainWindow(QMainWindow):
	def __init__(self):
		super().__init__()

		# Fit inside the 1280x1024 Xvfb used by start_browser_gui.sh
		self.xpos_main_window = 0
		self.ypos_main_window = 0
		self.width_main_window = 1280
		self.height_main_window = 960

		self.frame_duration_ms = 40

		self.half = 1

		# Defining some variables of the window
		self.title_main_window = "Event Annotator"

		# Setting the window appropriately
		self.setWindowTitle(self.title_main_window)
		self.setGeometry(self.xpos_main_window, self.ypos_main_window, self.width_main_window, self.height_main_window)
		self.setAutoFillBackground(True)

		# Initiate the sub-widgets
		self.init_main_window()

		# Show the window
		self.show()

	def init_main_window(self):

		# Add the media player
		self.media_player = MediaPlayer(self)
		video_display = QWidget(self)
		video_display.setAutoFillBackground(True)
		video_display.setLayout(self.media_player.layout)

		# Create the Event selection Window
		self.event_window = EventSelectionWindow(self)

		# Add the list
		self.list_display = ListDisplay(self)
		self.video_browser = VideoBrowser(self)

		# Create the original list of labels
		self.list_manager = ListManager()
		self.list_display.display_list(self.list_manager.create_text_list())


		# Layout the different widgets
		central_display = QWidget(self)
		central_display.setAutoFillBackground(True)
		self.setCentralWidget(central_display)

		help_label = QLabel(
			"Controls: Space play/pause · ←/→ frame · Enter annotate · Del delete · Ctrl+S save"
		)
		help_label.setStyleSheet("color: #cccccc; padding: 4px;")
		help_label.setWordWrap(True)

		right_column = QVBoxLayout()
		right_column.addWidget(self.video_browser, stretch=2)
		right_column.addWidget(self.list_display, stretch=1)
		right_column.addWidget(help_label)
		right_widget = QWidget(self)
		right_widget.setAutoFillBackground(True)
		right_widget.setLayout(right_column)

		final_layout = QHBoxLayout()
		final_layout.addWidget(video_display, stretch=3)
		final_layout.addWidget(right_widget, stretch=1)

		central_display.setLayout(final_layout)

		self.video_browser.refresh(auto_load_first=True)

	def keyPressEvent(self, event):

		ctrl = False

		# Remove an event with the delete key
		if event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
			index = self.list_display.list_widget.currentRow()
			if index >= 0:
				self.list_manager.delete_event(index)
				self.list_display.display_list(self.list_manager.create_text_list())
				path_label = self.media_player.get_last_label_file()
				self.list_manager.save_file(path_label, self.half)
			self.setFocus()

		# Play or pause the video with the space key
		if event.key() == Qt.Key_Space:
			if self.media_player.play_button.isEnabled():
				self.media_player.play_video()
				self.setFocus()

		# Move one frame backwards in time
		if event.key() == Qt.Key_Left:
			if self.media_player.play_button.isEnabled():
				position = self.media_player.media_player.position()
				if position > self.frame_duration_ms:
					self.media_player.media_player.setPosition(position-self.frame_duration_ms)
			self.setFocus()
		
		if event.key() == Qt.Key_Right:
			if self.media_player.play_button.isEnabled():
				position = self.media_player.media_player.position()
				duration = self.media_player.media_player.duration()
				if position < duration - self.frame_duration_ms:
					self.media_player.media_player.setPosition(position+self.frame_duration_ms)
			self.setFocus()

		# Enter a new annotation
		if event.key() == Qt.Key_Return:
			if self.media_player.play_button.isEnabled() and not self.media_player.media_player.state() == QMediaPlayer.PlayingState:	
				self.event_window.set_position()
				self.event_window.show()
				self.event_window.setFocus()
				self.event_window.list_widget.setFocus()
			self.setFocus()

		# Set the playback rate to normal
		if event.key() == Qt.Key_F1 or event.key() == Qt.Key_A:
			position = self.media_player.media_player.position()
			self.media_player.media_player.setPlaybackRate(1.0)
			self.media_player.media_player.setPosition(position)
			self.setFocus()

		# Set the playback rate to x2
		if event.key() == Qt.Key_F2 or event.key() == Qt.Key_Z:
			position = self.media_player.media_player.position()
			self.media_player.media_player.setPlaybackRate(2.0)
			self.media_player.media_player.setPosition(position)
			self.setFocus()

		# Set the playback rate to x4
		if event.key() == Qt.Key_F3 or event.key() == Qt.Key_E:
			position = self.media_player.media_player.position()
			self.media_player.media_player.setPlaybackRate(4.0)
			self.media_player.media_player.setPosition(position)
			self.setFocus()

		if event.key() == Qt.Key_Escape:
			self.list_display.list_widget.setCurrentRow(-1)
			self.setFocus()

		if event.modifiers() and Qt.ControlModifier:
			ctrl = True

		if event.key() == Qt.Key_S and ctrl:
			if self.media_player.play_button.isEnabled():
				path_label = self.media_player.get_last_label_file()
				self.list_manager.save_file(path_label, self.half)