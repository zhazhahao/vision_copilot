import numpy as np
import torch.multiprocessing as multiprocessing
from qinglang.utils.utils import Config
from qinglang.dataset.utils.utils import nms, xyxy2xywh
from qinglang.utils.io import clear_lines


class HandDetectorProcess(multiprocessing.Process):
    def __init__(self, init_done_barrier: multiprocessing.Barrier, inference_event: multiprocessing.Event, done_barrier: multiprocessing.Barrier, terminal_event: multiprocessing.Event, frame_shared_array: multiprocessing.Array, hand_detection_outputs: multiprocessing.Queue) -> None:
        super().__init__()
        self.init_done_barrier = init_done_barrier
        self.inference_event = inference_event
        self.done_barrier = done_barrier
        self.terminal_event = terminal_event
        self.frame_shared_array = frame_shared_array
        self.hand_detection_outputs = hand_detection_outputs
        self.daemon = True

    def init_process(self):
        self.config = Config('configs/hand_detection.yaml')
        self.source = Config('configs/source.yaml')
        
        from mmdeploy_runtime import Detector
        clear_lines(3)
        self.detector = Detector(model_path=self.source.onnx_path, device_name=self.config.device)
        clear_lines(1)

    def run(self) -> None:
        self.init_process()

        self.init_done_barrier.wait()
        
        while True:
            self.inference_event.wait()
            
            if self.terminal_event.is_set():
                break
            
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