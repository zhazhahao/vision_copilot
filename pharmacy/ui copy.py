import os, sys, glob
import numpy as np
from typing import Union
from PyQt6.QtOpenGLWidgets import QOpenGLWidget  
from PyQt6.QtWidgets import QMainWindow, QWidget, QGraphicsScene, QGraphicsView, QApplication, QFileDialog, QPushButton, QStyle, QHBoxLayout, QVBoxLayout, QLineEdit, QCheckBox, QSpacerItem, QSizePolicy, QFrame, QComboBox
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QSize, QEvent, QTimer, pyqtSignal
from PyQt6.QtGui import QPixmap, QAction, QPainter, QColor, QResizeEvent, QImage
from qinglang.dataset.coco.coco_base import COCOBase
from qinglang.dataset.coco.coco_toolbox import COCOToolbox
from qinglang.data_structure.image.image_base import ImageFlow
from qinglang.data_structure.video.video_toolbox import VideoToolbox
from qinglang.gui.utils import draw_hand_2d_on_qpixmap, draw_bbox_on_qpixmap, draw_bbox_category_on_qpixmap, signal_blocker
from qinglang.utils.utils import ClassDict, Config, Logger
from main import MainProcess

class UI(QMainWindow):
    def __init__(self, parent: Union[QWidget, None] = None) -> None:
        super().__init__(parent)
        
        self.logger = Logger()

        self.init_modules()        
        self.init_ui()

        self.status_module.ui_status.ready = True
        
        self.thread_module.monitor_thread.start()

    def init_modules(self) -> None:
        self.signal_module = SignalModule()
        self.status_module = StatusModule(self)
        self.thread_module = ThreadModule(self)

    def init_ui(self) -> None:
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        self.layout = QHBoxLayout()
        central_widget.setLayout(self.layout)

        self.init_display_module()

        self.setMinimumSize(320, 180)
        self.setWindowTitle('COCOViewer')
        self.show()
        
    def init_display_module(self):
        self.display_module = DisplayModule(self)
        self.layout.addWidget(self.display_module, 90)
    

class SignalModule(QObject):
    frameChecked_event = pyqtSignal(ClassDict)


class StatusModule:
    def __init__(self, parent: Union[QWidget, None] = None) -> None:
        self.ui_status = ClassDict(
            ready = False,
        )

        self.display_status = ClassDict(
            current_frame = QPixmap(1280, 720),
        )


class ThreadModule:
    def __init__(self, parent: Union[QWidget, None] = None) -> None:
        self.parent = parent

        self.monitor_thread = MonitorThread(self.parent)

class ImageProcessingThread(QThread):
    image_processed = pyqtSignal(QPixmap)  # 定义一个带参数的信号
    def __init__(self, results):
        super().__init__()
        self.results = results

    def run(self):
        pixmap = self.get_frame(self.results)
        self.image_processed.emit(pixmap)

    def get_frame(self, results):
        pixmap = QPixmap(QImage(results.frame.data, results.frame.shape[1], results.frame.shape[0], results.frame.strides[0], QImage.Format.Format_BGR888))
        scene = QGraphicsScene()
        self.plot_annotations(scene, pixmap, results)
        return pixmap

    def plot_annotations(self, scene, pixmap, results):
        for hand in results.hand_detection_results:
            self.plot_bbox(scene, pixmap, hand['bbox'], hand['category_id'])

        for drug in results.drug_detection_results:
            self.plot_bbox(scene, pixmap, drug['bbox'], drug['category_id'])

        for object_catched in results.check_results:
            self.plot_bbox(scene, pixmap, object_catched.get_latest_valid_node().bbox, object_catched.category_id, color=(0, 0, 255))

    def plot_bbox(self, scene, pixmap, bbox, category, color=(0, 255, 0)):
        draw_bbox_on_qpixmap(pixmap, np.array(bbox).astype(int), color)
        draw_bbox_category_on_qpixmap(pixmap, np.array(bbox).astype(int), category, color)

class DisplayModule(QWidget):
    def __init__(self, parent: Union[QWidget, None] = None) -> None:
        super().__init__(parent)
        
        self.parent = parent
        self.image_thread = None
 
        self._init_ui()
        self._init_signal_listener()

    def _init_ui(self) -> None:
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.layout.addWidget(self.view)

    def _init_signal_listener(self) -> None:
        self.parent.signal_module.frameChecked_event.connect(self.start_image_thread)

    def start_image_thread(self, results):
        if self.image_thread is None or not self.image_thread.isRunning():
            self.image_thread = ImageProcessingThread(results)
            self.image_thread.image_processed.connect(self.refresh_scene)  # 连接信号
            self.image_thread.start()

    def refresh_scene(self, pixmap):
        self.scene.clear()
        self.scene.addPixmap(pixmap)
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

class MonitorThread(QThread, MainProcess):
    def __init__(self, parent: Union[QWidget, None] = None):
        super().__init__()

        self.parent = parent

    def export_results(self, frame, check_results, hand_detection_results, drug_detection_results, hand_tracked, drug_tracked):
        self.parent.signal_module.frameChecked_event.emit(
            ClassDict(
                frame = frame,
                check_results = check_results,
                hand_detection_results = hand_detection_results,
                drug_detection_results = drug_detection_results,
                hand_tracked = hand_tracked,
                drug_tracked = drug_tracked,
            )
        )


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = UI()
    sys.exit(app.exec())
