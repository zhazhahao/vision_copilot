import sys
from  PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QObject, QDir
from PyQt6.QtQml import QQmlApplicationEngine

class Controller(QObject):
    def __init__(self):
        super().__init__()
        self._video_path = "/home/portable-00/data/video_0/20240313_160556.mp4"
        self._auto_play = True

    def getVideoPath(self):
        return self._video_path

    def isAutoPlay(self):
        return self._auto_play

if __name__ == "__main__":
    app = QApplication(sys.argv)
    controller = Controller()

    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("controller", controller)
    engine.load("/home/portable-00/VisionCopilot/pharmacy/test/qml_test.qml")

    sys.exit(app.exec())