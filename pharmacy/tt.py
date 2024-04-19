import sys
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtGui import QPainter, QColor


class MyWidget(QWidget):
    def __init__(self):
        super().__init__()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)  # 设置反走样
        painter.setBrush(QColor(255, 0, 0, 255))  # 设置半透明红色
        painter.drawRect(50, 50, 200, 100)  # 绘制矩形


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MyWidget()
    widget.resize(300, 200)
    widget.show()
    sys.exit(app.exec())
