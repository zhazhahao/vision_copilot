import numpy as np
import torch.multiprocessing as multiprocessing
from modules.ocr_detector import OcrDector


class OCRProcess(multiprocessing.Process):
    def __init__(self, inference_event: multiprocessing.Event, done_barrier: multiprocessing.Barrier, frame_shared_array: multiprocessing.Array, wild_ocr_outputs: multiprocessing.Queue) -> None:
        self.inference_event = inference_event
        self.done_barrier = done_barrier
        self.frame_shared_array = frame_shared_array
        self.wild_ocr_outputs = wild_ocr_outputs

        self.ocr_detector = OcrDector()

        super().__init__()

    def run(self):
        while True:
            self.inference_event.wait()
            self.execute()
    
    
    def execute(self) -> None:
        image = np.frombuffer(self.frame_shared_array.get_obj(), dtype=np.uint8).reshape((1080, 1920, 3))
        
        result = self.ocr_detector.ocr_detect(image)
        print(result)
        self.wild_ocr_outputs.put(result)
        self.done_barrier.wait()
        self.inference_event.clear()