import numpy as np
import torch.multiprocessing as multiprocessing
from typing import List, Dict
from modules.cameras import CameraBase
from modules.catch_checker import CatchChecker
from modules.drug_detector_process import DrugDetectorProcess
from modules.hand_detector_process import HandDetectorProcess
from modules.ocr_process import OCRProcess
from utils.utils import MedicineDatabase
from qinglang.utils.utils import ClassDict, Config, most_common


class Backbone:
    def __init__(self) -> None:
        multiprocessing.set_start_method('spawn')
 
        self.config = Config("configs/main.yaml", "configs/stream.yaml")
        self.source = Config("configs/source.yaml")

        self.medicine_database = MedicineDatabase()
        self.catch_checker = CatchChecker()
        
        self._init_shared_variables()
        self._init_subprocess()

        self.stream: CameraBase = ...
        self.prescription: List = ...

    def _init_shared_variables(self) -> None:
        self.frame_shared_array = multiprocessing.Array('B', int(np.prod(self.config.resolution)))

        self.ocr_event = multiprocessing.Event()
        self.ocr_done_barrier = multiprocessing.Barrier(2)

        self.detection_event = multiprocessing.Event()
        self.detection_done_barrier = multiprocessing.Barrier(3)
        
        self.hand_detection_queue = multiprocessing.Queue()
        self.drug_detection_queue = multiprocessing.Queue()
        self.ocr_queue = multiprocessing.Queue()

    def _init_subprocess(self) -> None:
        self.subprocesses = ClassDict(
            drug_detector = DrugDetectorProcess(self.detection_event, self.detection_done_barrier, self.frame_shared_array, self.drug_detection_queue),
            hand_detector = HandDetectorProcess(self.detection_event, self.detection_done_barrier, self.frame_shared_array, self.hand_detection_queue),
            ocr = OCRProcess(self.ocr_event, self.ocr_done_barrier, self.frame_shared_array, self.ocr_queue),
        )

        for p in self.subprocesses.values():
            p.start()

    def run(self) -> None:
        for frame in self.stream:
            self._share_frame_to_memory(frame)

            hand_detection_results, drug_detection_results, ocr_results = self._parallel_inference()

            if ocr_results != []:
                drug_detection_results = self._fiter_drug_detection_results(ocr_results, drug_detection_results)

            self.catch_checker.observe(hand_detection_results, drug_detection_results)
            check_results = self.catch_checker.check()

            self.post_process(frame, check_results, hand_detection_results, drug_detection_results, self.catch_checker.hand_tracker.tracked_objects, self.catch_checker.medicine_tracker.tracked_objects)

    def _share_frame_to_memory(self, frame: np.ndarray) -> None:
        np.copyto(np.frombuffer(self.frame_shared_array.get_obj(), dtype=np.uint8), frame.flatten())

    def _parallel_inference(self) -> None:
        # start parallel inference
        self.detection_event.set()
        self.ocr_event.set()
        self.detection_done_barrier.wait()
        self.ocr_done_barrier.wait()

        # finish parallel inference
        self.detection_event.clear()
        self.ocr_event.clear()
        self.detection_done_barrier.reset()
        self.ocr_done_barrier.reset()

        # get inference results
        hand_detection_results = self.hand_detection_queue.get()
        drug_detection_results = self.drug_detection_queue.get()
        ocr_results = self.ocr_queue.get()

        return hand_detection_results, drug_detection_results, ocr_results

    def _fiter_drug_detection_results(self, ocr_results: List[Dict], drug_detection_results: List[Dict]) -> List[Dict]:
        location = most_common([self.medicine_database[medicine['category_id']].get("Shelf") for medicine in ocr_results])
        return [drug for drug in drug_detection_results if self.medicine_database[drug['category_id']].get("Shelf") == location]

    def post_process(self, frame, check_results, hand_detection_results, drug_detection_results, hand_tracked, drug_tracked) -> None:
        ...


if __name__ == '__main__':
    from modules.cameras import VirtualCamera
    
    backbone = Backbone()
    backbone.stream = VirtualCamera(backbone.source.virtual_camera_source)

    backbone.run()
