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
from qinglang.data_structure.video.video_toolbox import VideoFlow, VideoToolbox
from qinglang.dataset.utils.utils import plot_xywh
from qinglang.utils.utils import Config, ClassDict, Logger, load_json, load_yaml

class VisualizeEngine(Engine):
    def __init__(self, history: ClassDict) -> None:
        super().__init__()
        
        self.history = history

    def post_process(self, frame, check_results, hand_detection_results, drug_detection_results, hand_tracked, drug_tracked):
        self.history.check_results.append([{'category_id': check_results[0].category_id, 'bbox': check_results[0].get_latest_valid_node().bbox}] if check_results != [] else [])
        self.history.hand_detection_results.append(hand_detection_results)
        self.history.drug_detection_results.append(drug_detection_results)


class PharmacyVisualizer:
    def __init__(self) -> None:
        self.config = Config("configs/benchmark.yaml")
        self.logger = Logger()

    def infer(self, dataset: ClassDict) -> None:
        history = ClassDict(
            check_results = [],
            hand_detection_results = [],
            drug_detection_results = [],
        )

        engine = VisualizeEngine(history)
        engine.stream = VirtualCamera(dataset.video)
        engine.prescription = load_yaml(dataset.prescription)
        
        engine.run()
        
        with open(os.path.join(dataset.root_path, 'results.pkl'), 'wb') as f:
            pickle.dump(history, f)

    def visualize(self, dataset):
        with open(os.path.join(dataset.root_path, 'results.pkl'), 'rb') as f:
            results = pickle.load(f)
        
        video_stream = VideoFlow("/home/portable-00/data/20240313_160556/20240313_160556.mp4")
        video_writer = VideoToolbox(video_stream).get_videoWriter("/home/portable-00/data/20240313_160556/visualized.mp4")
        
        for id, frame in enumerate(video_stream):
            if results.hand_detection_results[id] != []:
                for result in results.hand_detection_results[id]:
                    plot_xywh(frame, result['bbox'], category=result['category_id'])

            if results.drug_detection_results[id] != []:
                for result in results.drug_detection_results[id]:
                    plot_xywh(frame, result['bbox'], category=result['category_id'])
                    
            if results.check_results[id] != []:
                for result in results.check_results[id]:
                    # plot_xywh(frame, result.get_latest_valid_node().bbox, category=result.category_id, color=(0, 0, 255))
                    plot_xywh(frame, result['bbox'], category=result['category_id'], color=(0, 0, 255))
            
            video_writer.write(frame)
        video_writer.release()
        
        # video_toolbox = VideoToolbox("/home/portable-00/data/20240313_160556/visualized.mp4")
        # video_toolbox.to_images(path='visualized')


if __name__ == '__main__':
    benchmark = PharmacyVisualizer()
    benchmark.infer(ClassDict(benchmark.config.datasets))
    benchmark.visualize(ClassDict(benchmark.config.datasets))
    
    # import pickle
    # import numpy as np
    # from qinglang.utils.utils import ClassDict


    # with open("/home/portable-00/data/20240313_160556/full_results.pkl", 'rb') as f:
    #     loaded_list = pickle.load(f)
        
    # print(1)

    video_toolbox = VideoToolbox("/home/portable-00/data/20240313_160556/visualized.mp4")
    video_toolbox.to_images(path='visualized')