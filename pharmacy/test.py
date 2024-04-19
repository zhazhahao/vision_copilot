import sys
import random
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton
from main import MainProcess


class WorkerThread(QThread, MainProcess):
    result_ready = pyqtSignal(int)  # 定义一个信号，用于在处理完成时发送结果

    def __init__(self):
        # QThread.__init__(self)
        # MainProcess.__init__(self)
        super().__init__()

    def asdjaksdjaldjal(self):
        # 与window交互
        pass
    def update(self,frame,check_results):
        pass
    def terminate():
        QThread.terminate()
        MainProcess.terminate()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("QThread数据传输示例")
        self.setGeometry(100, 100, 300, 200)

        layout = QVBoxLayout()

        self.label = QLabel("等待结果...")
        layout.addWidget(self.label)

        self.button = QPushButton("开始处理")
        self.button.clicked.connect(self.start_processing)
        layout.addWidget(self.button)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.thread = WorkerThread()
        self.thread.result_ready.connect(self.handle_result)  # 连接信号到处理结果的方法

    def start_processing(self):
        self.button.setEnabled(False)  # 防止多次点击
        data = random.randint(0, 100)  # 生成一个随机数
        self.thread.data = data  # 将随机数传递给线程
        self.thread.start()

    def handle_result(self, result):
        self.label.setText("处理结果: {}".format(result))
        self.button.setEnabled(True)  # 处理完成后重新启用按钮

    def closeEvent(self, event):
        # 在窗口关闭时执行的函数
        print("MainWindow is closing")
        # 在这里可以执行你想要执行的代码
        self.thread.terminate()
        # event.accept()  # 确认关闭事件

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
