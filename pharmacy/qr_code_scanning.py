# import cv2
# import matplotlib.pyplot as plt

# # # 读取图像
# image = cv2.imread('qr_code_roi.png', cv2.IMREAD_GRAYSCALE)

# # 计算直方图
# histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

# # 绘制直方图
# plt.figure()
# plt.title("Grayscale Histogram")
# plt.xlabel("Bins")
# plt.ylabel("# of Pixels")
# plt.plot(histogram)
# plt.xlim([0, 256])
# plt.show()


# import cv2

# # 读取图像

# # 转换为灰度图像
# # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # 使用阈值 127 进行二值化
# ret, thresh = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY)

# # 显示结果
# cv2.imshow('thresh', thresh)
# cv2.waitKey(0)

import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QSlider
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage, qRgb
import cv2
import numpy as np

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Threshold Image")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(127)
        self.threshold_slider.valueChanged.connect(self.update_threshold)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.threshold_slider)

        self.central_widget.setLayout(layout)

        # Load initial image
        self.image = cv2.imread('qr_code_roi.png', cv2.IMREAD_GRAYSCALE)
        self.update_threshold()

    def update_threshold(self):
        threshold_value = self.threshold_slider.value()
        _, binary_image = cv2.threshold(self.image, threshold_value, 255, cv2.THRESH_BINARY)

        h, w = binary_image.shape
        bytes_per_line = w
        q_img = QImage(binary_image.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)

        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
