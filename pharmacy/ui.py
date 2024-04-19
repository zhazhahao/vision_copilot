import os, sys, glob
import numpy as np
from typing import Union
from PyQt6.QtWidgets import QMainWindow, QWidget, QGridLayout, QSlider, QLabel, QApplication, QFileDialog, QPushButton, QStyle, QHBoxLayout, QVBoxLayout, QLineEdit, QCheckBox, QSpacerItem, QSizePolicy, QFrame, QComboBox
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


class DisplayModule(QWidget):
    def __init__(self, parent: Union[QWidget, None] = None) -> None:
        super().__init__(parent)
        
        self.parent = parent
 
        self._init_ui()
        self._init_signal_listener()

    def _init_ui(self) -> None:
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self._init_display_panel()

    def _init_display_panel(self) -> None:
        self.display_panel = QLabel(self)
        self.display_panel.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        self.display_panel.setPixmap(self.parent.status_module.display_status.current_frame)
        self.layout.addWidget(self.display_panel)

    def _init_signal_listener(self) -> None:
        self.parent.signal_module.frameChecked_event.connect(self._set_frame)
        
    def _set_frame(self, results: ClassDict) -> None:
        self.parent.status_module.display_status.current_frame = self._get_frame(results)
        self.refresh_panel()

    def _get_frame(self, results: ClassDict) -> None:
        pixmap = QPixmap(QImage(results.frame.data, results.frame.shape[1], results.frame.shape[0], results.frame.strides[0], QImage.Format.Format_BGR888))
        
        self._plot_annotations(pixmap, results)
        
        return pixmap
    
    def _plot_annotations(self, pixmap: QPixmap, results: ClassDict) -> None:
        for hand in results.hand_detection_results:
            self._plot_bbox(pixmap, hand['bbox'], hand['category_id'])

        for drug in results.drug_detection_results:
            self._plot_bbox(pixmap, drug['bbox'], drug['category_id'])
                
        for object_catched in results.check_results:
            self._plot_bbox(pixmap, object_catched.get_latest_valid_node().bbox, object_catched.category_id, color=(0, 0, 255))

    def _plot_bbox(self, pixmap, bbox, category, color=(0, 255, 0)) -> None:
        draw_bbox_on_qpixmap(pixmap, np.array(bbox).astype(int), color)
        draw_bbox_category_on_qpixmap(pixmap, np.array(bbox).astype(int), category, color)
    
    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)

        if self.parent.status_module.ui_status.ready:
            self.refresh_panel()

    def refresh_panel(self) -> None:
        self.display_panel.setPixmap(self.parent.status_module.display_status.current_frame.scaled(self.display_panel.width(), self.display_panel.height(), aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio))


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
