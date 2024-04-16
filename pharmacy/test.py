import multiprocessing
import numpy as np
from typing import List, Dict
from mmdeploy_runtime import Detector
from qinglang.utils.utils import Config
from qinglang.hand_pose_estimation.nms import nms
from qinglang.dataset.utils.utils import xyxy2xywh
import subprocess

terminal_execute = lambda command: subprocess.run(command, shell=True, text=True).stdout

class HandDetector:
    def __init__(self) -> None:
        self.config = Config('configs/hand_detection.yaml')
        self.source = Config('configs/source.yaml')
        self.detector = Detector(model_path=self.source.onnx_path, device_name=self.config.device)
        
    def detect(self, image: np.ndarray) -> List[Dict[str, np.ndarray]]:
        bboxes, labels, _ = self.detector(image)
        bboxes = bboxes[np.logical_and(labels == self.config.class_id, bboxes[..., 4] > self.config.confidence_threshold)]
        bboxes = bboxes[nms(bboxes, 0.2)]
        return [{'bbox': xyxy2xywh(bbox[:4]), 'confidence': bbox[-1], 'category_id': 0} for bbox in bboxes]

def child_process(hand_detector):
    hand_detector
    from qinglang.data_structure.video.utils.utils import VideoFlow
    for frame in VideoFlow("/mnt/nas/datasets/Pharmacy_for_label/20240313/20240313_160556/20240313_160556.mp4"):
        results = hand_detector.detect(frame)
        print(results)

if __name__ == "__main__":
    hand_detector = HandDetector()
    # p = multiprocessing.Process(target=child_process, args=(hand_detector,))
    # p.start()
    # p.join()
    child_process(hand_detector)