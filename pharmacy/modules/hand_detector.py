import numpy as np
from typing import List, Dict
from mmdeploy_runtime import Detector
from qinglang.utils.utils import Config
from qinglang.hand_pose_estimation.nms import nms


class HandDetector:
    def __init__(self) -> None:
        self.config = Config('/home/portable-00/VisionCopilot/pharmacy/configs/hand_detection.yaml')
        self.detector = Detector(model_path=self.config.onnx_path, device_name=self.config.device)
        
    def detect(self, image: np.ndarray) -> List[Dict[str, np.ndarray]]:
        bboxes, labels, _ = self.detector(image)
        bboxes = bboxes[np.logical_and(labels == self.config.class_id, bboxes[..., 4] > self.config.confidence_threshold)]
        bboxes = bboxes[nms(bboxes, 0.2)]
        return [{'bbox': bbox[:4], 'confidence': bbox[-1], 'category_id': labels} for bbox in bboxes]


if __name__ == '__main__':
    ...