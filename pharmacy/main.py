import os
import time
import cv2
import sys
import signal
import multiprocessing
import numpy as np
from typing import Union, List, Dict, Any
from modules.catch_checker import CatchChecker
from pharmacy.modules.drug_detector_process import DrugDetectorProcess
from modules.hand_detector_process import HandDetectorProcess
from modules.cameras import VirtualCamera
from qinglang.dataset.utils.utils import plot_xywh, centerwh2xywh
from qinglang.utils.utils import ClassDict


class MainProcess:
    def __init__(self) -> None:
        self.stream = VirtualCamera()

        self.init_shared_variables()
        self.init_subprocess()

        self.catch_checker = CatchChecker()
    
    def init_shared_variables(self):
        self.inference_event = multiprocessing.Event()
        self.frame_inputs = multiprocessing.Queue()
        self.hand_detection_outputs = multiprocessing.Queue()
        self.drug_detection_outputs = multiprocessing.Queue()

    def init_subprocess(self) -> None:
        self.subprocesses = ClassDict(
            drug_detector = DrugDetectorProcess(self.inference_event),
            hand_detector = HandDetectorProcess(self.inference_event),
        )

        for p in self.subprocesses.values():
            p.start()

    def run(self):
        for frame in self.stream:
            self.share_frame(frame)

            for p in self.subprocesses.values():
                os.kill(p.pid, signal.SIGUSR1)

            self.inference_event.wait()

            hand_detection_results = self.hand_detection_outputs.get()
            drug_detection_results = self.drug_detection_outputs.get()

            self.catch_checker.observe(hand_detection_results, drug_detection_results)
            check_results = self.catch_checker.check()
    
    def share_frame(self, frame: np.ndarray) -> None:
        ...
        
if __name__ == '__main__':
    test1 = MainProcess()
    test1.run()
