import numpy as np
import torch.multiprocessing as multiprocessing
from mmdeploy_runtime import Detector
from qinglang.utils.utils import Config
from qinglang.dataset.utils.utils import nms, xyxy2xywh


class HandDetectorProcess(multiprocessing.Process):
    def __init__(self, inference_event: multiprocessing.Event, done_barrier: multiprocessing.Barrier, frame_shared_array: multiprocessing.Array, hand_detection_outputs: multiprocessing.Queue) -> None:
        
        self.inference_event = inference_event
        self.done_barrier = done_barrier
        self.frame_shared_array = frame_shared_array
        self.hand_detection_outputs = hand_detection_outputs

        super().__init__()
    
    def init_process(self):
        self.config = Config('configs/hand_detection.yaml')
        self.source = Config('configs/source.yaml')
        self.detector = Detector(model_path=self.source.onnx_path, device_name=self.config.device)

    def run(self) -> None:
        self.init_process()

        while True:
            self.inference_event.wait()
            self.execute()
    
    def execute(self) -> None:
        image = np.frombuffer(self.frame_shared_array.get_obj(), dtype=np.uint8).reshape((1080, 1920, 3))
        bboxes, labels, _ = self.detector(image)
        bboxes = bboxes[np.logical_and(labels == self.config.class_id, bboxes[..., 4] > self.config.confidence_threshold)]
        bboxes = bboxes[nms(bboxes, self.config.overlap_threshold)]
        hand_detection_result = [{'bbox': xyxy2xywh(bbox[:4]), 'category_id': 0} for bbox in bboxes]
        self.hand_detection_outputs.put(hand_detection_result)
        
        self.done_barrier.wait()
        self.inference_event.clear()