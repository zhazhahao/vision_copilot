import cv2
import numpy as np
from typing import List, Dict
from mmdeploy_runtime import Detector
from qinglang.utils.utils import Config
from qinglang.hand_pose_estimation.nms import nms
from qinglang.dataset.utils.utils import xyxy2xywh
from qinglang.dataset.utils.utils import plot_xyxy
import os, sys

sys.path.append('/home/portable-00/VisionCopilot/pharmacy/')

class HandDetector:
    def __init__(self) -> None:
        self.config = Config('/home/portable-00/VisionCopilot/pharmacy/configs/hand_detection.yaml')
        self.detector = Detector(model_path=self.config.onnx_path, device_name=self.config.device)
        
    def detect(self, image: np.ndarray) -> List[Dict[str, np.ndarray]]:
        bboxes, labels, _ = self.detector(image)
        bboxes = bboxes[np.logical_and(labels == self.config.class_id, bboxes[..., 4] > self.config.confidence_threshold)]
        bboxes = bboxes[nms(bboxes, 0.2)]
        return [{'bbox': xyxy2xywh(bbox[:4]), 'confidence': bbox[-1], 'category_id': 0} for bbox in bboxes]


if __name__ == '__main__':
    from camera_processor import CameraProcessor
    hand_detector = HandDetector()
    cam_stream = CameraProcessor()
    hand_metainfo = Config(rf'/home/portable-00/lab/perception/qinglang/data_structure/hand/metainfo/COCO WholeBody.yaml')

    while True:
        valid, frame = cam_stream.achieve_image()
        if not valid:
            continue

        results = hand_detector.detect(frame)
        for result in results:
            plot_xyxy(frame, np.array(result['bbox'], dtype=int))
        cv2.imshow("img",frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break