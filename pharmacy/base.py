import os
import json
import cv2
import time
import pickle
import numpy as np
from itertools import groupby
from typing import Dict, List
from ultralytics import YOLO
from engine import Engine
from modules.cameras import VirtualCamera
from qinglang.utils.utils import Config, ClassDict, Logger, load_json, load_yaml


class BaseEngine(Engine):
    def __init__(self) -> None:
        super().__init__()
        
    def fore_process(self):
        ...

    def post_process(self, frame, check_results, hand_detection_results, drug_detection_results, hand_tracked, drug_tracked):
        ...
        

class PharmacyCopilotBase:
    def __init__(self) -> None:
        self.config = Config("configs/benchmark.yaml")
        self.logger = Logger()

    def run(self, stream_src: VirtualCamera) -> None:


        engine = BaseEngine()
        engine.stream = stream_src

        engine.run()


if __name__ == '__main__':
    base = PharmacyCopilotBase()
    base.run(VirtualCamera("/home/portable-00/data/datasets/20240313_160556/20240313_160556.mp4"))
