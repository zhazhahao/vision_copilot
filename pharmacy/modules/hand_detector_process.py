import multiprocessing
import numpy as np
from mmdeploy_runtime import Detector
from qinglang.utils.utils import Config
from qinglang.dataset.utils.utils import nms, xyxy2xywh
import cv2


class HandDetectorProcess(multiprocessing.Process):
    def __init__(self, inference_event: multiprocessing.Event, done_barrier: multiprocessing.Barrier, frame_shared_array: multiprocessing.Array, hand_detection_outputs: multiprocessing.Queue) -> None:
        self.inference_event = inference_event
        self.done_barrier = done_barrier
        self.frame_shared_array = frame_shared_array
        self.hand_detection_outputs = hand_detection_outputs
        
        self.config = Config('configs/hand_detection.yaml')
        self.source = Config('configs/source.yaml')
        self.detector = Detector(model_path=self.source.onnx_path, device_name=self.config.device)

        super().__init__()

    def run(self) -> None:
        while True:
            self.inference_event.wait()
            self.execute()
    
    def execute(self) -> None:
        image_ = np.frombuffer(self.frame_shared_array.get_obj(), dtype=np.uint8).reshape((1080, 1920, 3))
        image = cv2.imread("/home/qinglang/lab/VisionCopilot/pharmacy/image.png")
        bboxes, labels, _ = self.detector(image)
        bboxes = bboxes[np.logical_and(labels == self.config.class_id, bboxes[..., 4] > self.config.confidence_threshold)]
        bboxes = bboxes[nms(bboxes, self.config.overlap_threshold)]
        hand_detection_result = [{'bbox': xyxy2xywh(bbox[:4]), 'category_id': 0} for bbox in bboxes]
        self.hand_detection_outputs.put(hand_detection_result)
        
        self.done_barrier.wait()
        self.inference_event.clear()