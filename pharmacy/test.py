import sys
import random
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton
from main import MainProcess
from qinglang.utils.utils import ClassDict

class WorkerThread(QThread, MainProcess):
    result_ready = pyqtSignal(ClassDict)  # 定义一个信号，用于在处理完成时发送结果

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        print(self.parent)

    def export_results(self, frame, check_results, hand_detection_results, drug_detection_results, hand_tracked, drug_tracked):
        self.result_ready.emit(
            ClassDict(
                frame = frame,
                check_results = check_results,
                hand_detection_results = hand_detection_results,
                drug_detection_results = drug_detection_results,
                hand_tracked = hand_tracked,
                drug_tracked = drug_tracked,
            )
        )



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

        self.thread = WorkerThread(self)
        self.thread.result_ready.connect(self.handle_result)  # 连接信号到处理结果的方法

    def start_processing(self):
        self.button.setEnabled(False)  # 防止多次点击
        self.thread.start()

    def handle_result(self, result):
        msg = (
            "---------------------------------------------------------------------" "\n"
            rf"{result.check_results}" "\n"
            rf"{result.hand_detection_results}" "\n"
            rf"{result.drug_detection_results}" "\n"
            rf"{result.hand_tracked}" "\n"
            rf"{result.drug_tracked}" "\n"
        )

        self.label.setText(msg)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
