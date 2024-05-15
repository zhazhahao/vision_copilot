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
    def __init__(self, history: ClassDict) -> None:
        super().__init__()
        
        self.frame_id = 0
        self.history = history
        
    def fore_process(self):
        ...

    def post_process(self, frame, check_results, hand_detection_results, drug_detection_results, hand_tracked, drug_tracked):
        self.history.check_results.append(check_results)
        self.history.hand_detection_results.append(hand_detection_results)
        self.history.drug_detection_results.append(drug_detection_results)
        
        self.frame_id += 1


class PharmacyCopilotBase:
    def __init__(self) -> None:
        self.config = Config("configs/benchmark.yaml")
        self.logger = Logger()

    def run(self, dataset: ClassDict) -> None:
        history = ClassDict(
            check_results = [],
            hand_detection_results = [],
            drug_detection_results = [],
        )

        engine = BaseEngine(history)
        engine.stream = VirtualCamera(dataset.video)

        engine.run()
        
        with open(os.path.join(dataset.root_path, '20240514.pkl'), 'wb') as f:
            pickle.dump(history, f)

if __name__ == '__main__':
    base = PharmacyCopilotBase()
    base.run(ClassDict(base.config.datasets))

    from qinglang.data_structure.video.video_toolbox import VideoToolbox
    video_toolbox = VideoToolbox("/home/portable-00/data/20240313_160556/visualized.mp4")
    video_toolbox.to_images(path='visualized')