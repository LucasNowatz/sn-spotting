import os
import sys

# Headless Xvfb: force software rendering so widgets paint into the framebuffer.
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("QT_X11_NO_MITSHM", "1")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("QT_OPENGL", "software")

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QApplication

from interface.main_window import MainWindow


def _apply_fusion_theme(app: QApplication) -> None:
	app.setStyle("Fusion")
	palette = QPalette()
	bg = QColor(45, 45, 48)
	fg = QColor(240, 240, 240)
	palette.setColor(QPalette.Window, bg)
	palette.setColor(QPalette.WindowText, fg)
	palette.setColor(QPalette.Base, QColor(30, 30, 30))
	palette.setColor(QPalette.AlternateBase, bg)
	palette.setColor(QPalette.Text, fg)
	palette.setColor(QPalette.Button, QColor(62, 62, 66))
	palette.setColor(QPalette.ButtonText, fg)
	palette.setColor(QPalette.Highlight, QColor(0, 122, 204))
	palette.setColor(QPalette.HighlightedText, Qt.white)
	app.setPalette(palette)


if __name__ == "__main__":
	QApplication.setAttribute(Qt.AA_UseSoftwareOpenGL, True)
	QApplication.setAttribute(Qt.AA_ForceRasterWidgets, True)

	application = QApplication(sys.argv)
	_apply_fusion_theme(application)
	window = MainWindow()
	sys.exit(application.exec_())
