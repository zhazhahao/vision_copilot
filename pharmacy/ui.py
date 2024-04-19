import os, sys, glob
import numpy as np
from typing import Union
from PyQt6.QtWidgets import QMainWindow, QWidget, QGridLayout, QSlider, QLabel, QApplication, QFileDialog, QPushButton, QStyle, QHBoxLayout, QVBoxLayout, QLineEdit, QCheckBox, QSpacerItem, QSizePolicy, QFrame, QComboBox
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QSize, QEvent, QTimer, pyqtSignal
from PyQt6.QtGui import QPixmap, QAction, QPainter, QColor, QResizeEvent
from qinglang.dataset.coco.coco_base import COCOBase
from qinglang.dataset.coco.coco_toolbox import COCOToolbox
from qinglang.data_structure.image.image_base import ImageFlow
from qinglang.data_structure.video.video_toolbox import VideoToolbox
from qinglang.gui.utils import draw_hand_2d_on_qpixmap, draw_bbox_on_qpixmap, draw_bbox_confidence_on_qpixmap, signal_blocker
from qinglang.utils.utils import ClassDict, Config, Logger


class UI(QMainWindow):
    def __init__(self, parent: Union[QWidget, None] = None) -> None:
        super().__init__(parent)
        
        self.logger = Logger()

        self.init_modules()        
        self.init_ui()

        self.status_module.ui_status.ready = True

    def init_modules(self) -> None:
        self.data_module = DataModule(self)
        self.status_module = StatusModule(self)
        self.config_module = ConfigModule(self)
        self.thread_module = ThreadModule(self)
        self.signal_module = SignalModule()

    def init_ui(self) -> None:
        # Main layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        self.layout = QHBoxLayout()
        central_widget.setLayout(self.layout)

        self.init_display_module()
        self.init_side_panel()
        self.init_menu_bar()

        self.setMinimumSize(320, 180)
        self.setWindowTitle('COCOViewer')
        self.show()
        
    def init_display_module(self):
        self.display_module = DisplayModule(self)
        self.layout.addWidget(self.display_module, 90)
    
    def init_side_panel(self):
        layout = QVBoxLayout()
        
        # Setting module
        self.setting_module = SettingModule(self)
        layout.addWidget(self.setting_module)

        layout.addStretch(1)
        
        # Function module
        self.function_module = FunctionModule(self)
        layout.addWidget(self.function_module)

        self.layout.addLayout(layout, 15)
                
    def init_menu_bar(self):
        file_menu = self.menuBar().addMenu('File')
        
        laod_data_action = QAction('Open Folder', self)
        laod_data_action.triggered.connect(self.load_dataset)
        laod_data_action.setShortcut('Ctrl+O')
        file_menu.addAction(laod_data_action)

    def load_dataset(self) -> None:
        # root_path = QFileDialog.getExistingDirectory(self, "Select Dataset", "/mnt/nas/datasets/Pharmacy")
        root_path = QFileDialog.getExistingDirectory(self, "Select Dataset", "C:/Users/Geniu/Desktop/test")
        self.data_module.load(root_path)
        self.display_module.reset()


class DataModule:
    def __init__(self, parent: Union[QWidget, None] = None) -> None:
        self.parent = parent
        self._reset()
    
    def load(self, path: str):
        self._reset()
        self.metainfo.root_path = path

        self._preprocess()

        self._load_images()
        self._load_annotations()

    def _preprocess(self):
        file_list = os.listdir(self.metainfo.root_path)
        
        if 'images' not in file_list:
            video_list = [file for file in file_list if file.lower().endswith(('.mp4', '.avi'))]

            if len(video_list) == 1:
                self.parent.logger.debug(rf"Valid dataset with no accessible images found.")
                VideoToolbox(os.path.join(self.metainfo.root_path, video_list[0])).to_images()
            else:
                self.parent.logger.debug(rf"Dataset loading error, please check data architecture.")

    def _load_images(self):
        self.images = ImageFlow(os.path.join(self.metainfo.root_path, 'images'), return_as='path')

    def _load_annotations(self):
        self.annotations = {os.path.basename(json_path).replace('.json', ''): None for json_path in sorted(glob.glob(os.path.join(self.metainfo.root_path, '*.json')))}

        self.parent.setting_module.reload_dataset_info()

    def _reset(self):
        self.__dict__ = {'parent': self.parent}

        self.metainfo = ClassDict()

    def activate_annotation(self, annotation_name: str) -> None:
        if annotation_name not in self.annotations or self.annotations[annotation_name] is None:
            coco = COCOBase(self.metainfo.root_path, rf'{annotation_name}.json')
            coco.images.return_as('path')
            self.annotations[annotation_name] = coco

        self.parent.status_module.dataset_status.current_annotation = self.annotations[annotation_name]


class StatusModule:
    def __init__(self, parent: Union[QWidget, None] = None) -> None:
        self.ui_status = ClassDict(
            ready = False,
        )

        self.dataset_status = ClassDict(
            current_annotation = None,
        )

        self.display_status = ClassDict(
            current_frame = QPixmap(1280, 720),
            current_frame_id = None,
        )

        self.export_frame_status = ClassDict(
            frame_list = [],
        )


class ConfigModule:
    def __init__(self, parent: Union[QWidget, None] = None) -> None:
        self.display_config = ClassDict(
            fps = 30,
            bboxes_colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 255)],
            bboxes_threshold = 1,
            display_bboxes_confidence = True,
            joints_threshold = 1,
            display_joints_confidence = True,
            # hand_metainfo = Config(rf'/home/qinglang/lab/codes/qinglang/data_structure/hand/metainfo/COCO WholeBody.yaml'),
        )


class SignalModule(QObject):
    setFrame_event = pyqtSignal()


class ThreadModule:
    def __init__(self, parent: Union[QWidget, None] = None) -> None:
        self.parent = parent

        self.autoPlay_thread = None


class DisplayModule(QWidget):
    def __init__(self, parent: Union[QWidget, None] = None) -> None:
        super().__init__(parent)
        
        self.parent = parent
 
        self._init_ui()
        self._init_threads()

    def _init_ui(self) -> None:
        # Parent layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self._init_display_panel()
    
    def _init_display_panel(self) -> None:
        self.display_panel = QLabel(self)
        self.display_panel.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        self.display_panel.setPixmap(self.parent.status_module.display_status.current_frame)
        self.layout.addWidget(self.display_panel)

    def _set_frame(self, value: int) -> None:
        self.parent.status_module.display_status.current_frame_id = value
        self.parent.status_module.display_status.current_frame = self._get_frame()
        self.refresh_panel()

        self.parent.logger.debug(rf"Move to frame {self.parent.status_module.display_status.current_frame_id}")
        self.parent.signal_module.setFrame_event.emit()

    def _get_frame(self) -> None:
        pixmap = QPixmap(self.parent.data_module.images[self.parent.status_module.display_status.current_frame_id])
        
        if self.parent.status_module.dataset_status.current_annotation != None:
            self._plot_annotations(pixmap)
        
        return pixmap
    
    def _plot_annotations(self, pixmap: QPixmap) -> None:
        for annotation in [self.parent.status_module.dataset_status.current_annotation.annotations.annotations[id] for id in self.parent.status_module.dataset_status.current_annotation.annotations.images[self.parent.status_module.display_status.current_frame_id]['annotation_ids']]:

            # if self.parent.config_module.display_config.joints_threshold != 1:
            #     self._plot_hands(pixmap, annotations)
            
            if annotation['bbox_confidence'] >= self.parent.config_module.display_config.bboxes_threshold:
                self._plot_bbox(pixmap, annotation)

    def _plot_hands(self, pixmap, annotations) -> None:
        [draw_hand_2d_on_qpixmap(pixmap, np.array(annotation['keypoints']).astype(int), self.parent.config_module.display_config.hand_metainfo) for annotation in annotations if np.mean(annotation['keypoints_confidence']) > self.parent.config_module.display_config.joints_threshold]

    def _plot_bbox(self, pixmap, annotation, color=(0, 255, 0)) -> None:
        draw_bbox_on_qpixmap(pixmap, np.array(annotation['bbox']).astype(int), color)

        if self.parent.config_module.display_config.display_bboxes_confidence == True:
            draw_bbox_confidence_on_qpixmap(pixmap, np.array(annotation['bbox']).astype(int), np.round(np.array(annotation['bbox_confidence']), 2))

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)

        if self.parent.status_module.ui_status.ready:
            self.refresh_panel()

    def refresh_panel(self) -> None:
        self.display_panel.setPixmap(self.parent.status_module.display_status.current_frame.scaled(self.display_panel.width(), self.display_panel.height(), aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio))

    def reset(self) -> None:
        self.display_slider.setMaximum(len(self.parent.data_module.images) - 1)
        self._set_frame(0)
        self.autoPlay_button.setEnabled(True)
        self.display_slider.setEnabled(True)


class SettingModule(QWidget):
    def __init__(self, parent: Union[QWidget, None] = None) -> None:
        super().__init__(parent)
        
        self.parent = parent

        self._init_ui()
    
    def _init_ui(self) -> None:
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self._init_dataset_setting_panel()
        self._add_separator()
        self._init_bboxes_setting_panel()
        self._add_separator()
        self._init_joints_setting_panel()

        self.bboxesThreshold_lineEdit.setText('0')
        self.bboxesThreshold_slider.setValue(0)
        self.jointsThreshold_lineEdit.setText('0')
        self.jointsThreshold_slider.setValue(0)


    def _init_dataset_setting_panel(self):
        layout = QVBoxLayout()

        datasetSetting_label = QLabel("Dataset Settings")
        datasetSetting_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(datasetSetting_label)

        self.annotation_comboBox = QComboBox(self)
        self.annotation_comboBox.setEditable(True)
        self.annotation_comboBox.textActivated[str].connect(self.parent.data_module.activate_annotation)
        layout.addWidget(self.annotation_comboBox)
            
        self.layout.addLayout(layout)

    def _init_bboxes_setting_panel(self):
        layout = QGridLayout()

        bboxesSetting_label = QLabel("Bboxes Settings")
        bboxesSetting_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(bboxesSetting_label, 0, 0, 1, 4)

        self.display_bboxesConfidence_checkBox = QCheckBox('Display Confidence', self)
        self.display_bboxesConfidence_checkBox.toggle()
        self.display_bboxesConfidence_checkBox.stateChanged.connect(self.switch_display_bboxes_confidence)
        layout.addWidget(self.display_bboxesConfidence_checkBox, 1, 0, 1, 4)

        bboxesThreshold_label = QLabel("Threshold: ")
        layout.addWidget(bboxesThreshold_label, 2, 0, 1, 3)

        self.bboxesThreshold_lineEdit = QLineEdit()
        self.bboxesThreshold_lineEdit.textChanged.connect(self.bboxesThreshold_lineEdit_changed)
        layout.addWidget(self.bboxesThreshold_lineEdit, 2, 3, 1, 1)

        self.bboxesThreshold_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.bboxesThreshold_slider.setMaximum(100)
        self.bboxesThreshold_slider.valueChanged.connect(self.bboxesThreshold_slider_changed)
        layout.addWidget(self.bboxesThreshold_slider, 3, 0, 1, 4)

        self.layout.addLayout(layout)

    def _init_joints_setting_panel(self):
        layout = QGridLayout()

        jointsSetting_label = QLabel("Joints Settings")
        jointsSetting_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(jointsSetting_label, 0, 0, 1, 4)

        self.display_jointsConfidence_checkBox = QCheckBox('Display Confidence', self)
        self.display_jointsConfidence_checkBox.toggle()
        self.display_jointsConfidence_checkBox.stateChanged.connect(self.switch_displayJointsConfidence)
        layout.addWidget(self.display_jointsConfidence_checkBox, 1, 0, 1, 4)

        jointsThreshold_label = QLabel("Threshold: ")
        layout.addWidget(jointsThreshold_label, 2, 0, 1, 3)

        self.jointsThreshold_lineEdit = QLineEdit()
        self.jointsThreshold_lineEdit.textChanged.connect(self.joints_threshold_lineEdit_changed)
        layout.addWidget(self.jointsThreshold_lineEdit, 2, 3, 1, 1)

        self.jointsThreshold_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.jointsThreshold_slider.setMaximum(100)
        self.jointsThreshold_slider.valueChanged.connect(self.joints_threshold_slider_changed)
        layout.addWidget(self.jointsThreshold_slider, 3, 0, 1, 4)

        self.layout.addLayout(layout)
    
    def _add_separator(self):
        layout = QVBoxLayout()

        layout.addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator)

        layout.addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        
        self.layout.addLayout(layout)

    def reload_dataset_info(self):
        self.annotation_comboBox.clear()

        for annotation_name in self.parent.data_module.annotations:
            self.annotation_comboBox.addItem(annotation_name)
    
    @signal_blocker('bboxesThreshold_slider')
    def bboxesThreshold_lineEdit_changed(self, text: str) -> None:
        try:
            if 0 <= (value := float(text)) <= 1:
                self.bboxesThreshold_slider.setValue(int(value * 100))
                self.set_bboxes_threshold(value)
        except ValueError:
            pass
        
    @signal_blocker('bboxesThreshold_lineEdit')
    def bboxesThreshold_slider_changed(self, value: int) -> None:
        self.bboxesThreshold_lineEdit.setText(str(value / 100))
        self.set_bboxes_threshold(value / 100)   
            
    def set_bboxes_threshold(self, value: float) -> None:
        self.parent.config_module.display_config.bboxes_threshold = value
        self.parent.logger.debug(rf"BBoxes confidence threshold set to {value}.")
        
    @signal_blocker('jointsThreshold_slider')
    def joints_threshold_lineEdit_changed(self, text: str) -> None:
        try:
            if 0 <= (value := float(text)) <= 1:
                self.jointsThreshold_slider.setValue(int(value * 100))
                self.set_joints_threshold(value)
        except ValueError:
            pass
            
    @signal_blocker('jointsThreshold_lineEdit')
    def joints_threshold_slider_changed(self, value: int) -> None:
        self.jointsThreshold_lineEdit.setText(str(value / 100))
        self.set_joints_threshold(value / 100)   
            
    def set_joints_threshold(self, value: float) -> None:
        self.parent.config_module.display_config.joints_threshold = value
        self.parent.logger.debug(rf"Joints confidence threshold set to {value}.")

    def switch_display_bboxes_confidence(self, state) -> None:
        status = state == Qt.CheckState.Checked.value
        self.parent.config_module.display_config.display_bboxes_confidence = status
        self.parent.logger.debug(rf"Plot bboxes confidence: {status}.")

    def switch_displayJointsConfidence(self, state) -> None:
        status = state == Qt.CheckState.Checked.value
        self.parent.config_module.display_config.display_joints_confidence = status
        self.parent.logger.debug(rf"Plot joints confidence: {status}.")


class FunctionModule(QWidget):
    def __init__(self, parent: Union[QWidget, None] = None) -> None:
        super().__init__(parent)
        
        self.parent = parent
                
        self._init_ui()
        self.init_listener()
    
    def _init_ui(self) -> None:
        # Grid layout
        layout = QGridLayout()
        self.setLayout(layout)
        
        # Export frame button
        self.exportFrame_button = QPushButton('Export Frame')
        self.exportFrame_button.setShortcut('S')
        self.exportFrame_button.setFixedHeight(36)
        self.exportFrame_button.clicked.connect(self.export_frame)
        layout.addWidget(self.exportFrame_button, 0, 0, 1, 1)

    def init_listener(self) -> None:
        self.parent.signal_module.setFrame_event.connect(self.check_frame_status)
        
    def export_frame(self):
        if self.parent.status_module.export_frame_status.frame_list == []:
            self.parent.data_module.coco_selected_toolbox = COCOToolbox(COCOBase(self.parent.data_module.cocos[self.parent.config_module.dataset_config.main_id].metainfo.path, 'coco_selected.json'))
            self.parent.data_module.coco_selected_toolbox.coco.annotations.categories = self.parent.data_module.cocos[self.parent.config_module.dataset_config.main_id].annotations.categories

        data = {
            'image': self.parent.data_module.cocos[self.parent.config_module.dataset_config.main_id].annotations.images[self.parent.status_module.display_status.current_frame_id],
            'annotations': [self.parent.data_module.cocos[self.parent.config_module.dataset_config.main_id].annotations.annotations[id] for id in self.parent.data_module.cocos[self.parent.config_module.dataset_config.main_id].annotations.images[self.parent.status_module.display_status.current_frame_id]['annotation_ids']],
        }
        self.parent.data_module.coco_selected_toolbox.export_frame_data(data)
        self.parent.data_module.coco_selected_toolbox.save()

        self.parent.logger.debug(rf"Export frame {self.parent.status_module.display_status.current_frame_id} to coco_selected.json")

        self.parent.status_module.export_frame_status.frame_list.append(self.parent.status_module.display_status.current_frame_id)
        self.check_frame_status()
    
    def check_frame_status(self):
        if self.parent.status_module.display_status.current_frame_id in self.parent.status_module.export_frame_status.frame_list:
            self.exportFrame_button.setText('Frame Exported')
            self.exportFrame_button.setEnabled(False)
        else:
            self.exportFrame_button.setText('Export Frame')
            self.exportFrame_button.setEnabled(True)
            self.exportFrame_button.setShortcut('S')
            

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = UI()
    sys.exit(app.exec())
