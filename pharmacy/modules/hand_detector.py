import cv2
import numpy as np
from typing import List, Dict
from mmdeploy_runtime import Detector
from qinglang.utils.utils import Config
from qinglang.hand_pose_estimation.nms import nms
from qinglang.dataset.utils.utils import xyxy2xywh
from qinglang.dataset.utils.utils import plot_xywh
import os, sys


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
    hand_metainfo = Config(rf'/home/portable-00/lab/perception/qinglang/data_structure/hand/metainfo/COCO WholeBody.yaml')
    from qinglang.data_structure.video.utils.utils import VideoFlow

    hand_detector = HandDetector()

    for frame in VideoFlow("/home/portable-00/VisionCopilot/pharmacy/20240313_160556/20240313_160556.mp4"):
        results = hand_detector.detect(frame)
        for result in results:
            plot_xywh(frame, np.array(result['bbox'], dtype=int))
        
        cv2.imshow('frame', frame)
        cv2.waitKey(10)
        
    
    # from camera_processor import CameraProcessor
    # cam_stream = CameraProcessor()

    # while True:
    #     valid, frame = cam_stream.achieve_image()
    #     if not valid:
    #         continue

    #     results = hand_detector.detect(frame)
    #     for result in results:
    #         plot_xyxy(frame, np.array(result['bbox'], dtype=int))
    #     cv2.imshow("img",frame)

    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break