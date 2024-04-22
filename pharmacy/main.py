import cv2
import numpy as np
import torch.multiprocessing as multiprocessing
from copy import deepcopy
from datetime import datetime
from modules.cameras import VirtualCamera
from modules.catch_checker import CatchChecker
from modules.drug_detector_process import DrugDetectorProcess
from modules.hand_detector_process import HandDetectorProcess
from modules.wild_ocr_process import WildOcrDecetor
from qinglang.utils.utils import ClassDict, Config
from qinglang.dataset.utils.utils import plot_xywh


class MainProcess:
    def __init__(self) -> None:
        multiprocessing.set_start_method('spawn')
 
        self.config = Config("configs/main.yaml")
        self.source = Config("configs/source.yaml")
        
        self.stream = VirtualCamera(self.source.virtual_camera_source)
        self.init_shared_variables()
        self.init_subprocess()
        self.catch_checker = CatchChecker()
        t = datetime.now().strftime(rf"%Y%m%d-%H%M%S")
        self.work_dir = rf"work_dirs/{t}"
    
    def init_shared_variables(self):
        self.frame_shared_array = multiprocessing.Array('B', 1920 * 1080 * 3)
        self.inference_event = multiprocessing.Event()
        self.done_barrier = multiprocessing.Barrier(4)
        self.hand_detection_queue = multiprocessing.Queue()
        self.drug_detection_queue = multiprocessing.Queue()
        self.wild_ocr_queue = multiprocessing.Queue()

    def init_subprocess(self) -> None:
        self.subprocesses = ClassDict(
            drug_detector = DrugDetectorProcess(self.inference_event, self.done_barrier, self.frame_shared_array, self.drug_detection_queue),
            hand_detector = HandDetectorProcess(self.inference_event, self.done_barrier, self.frame_shared_array, self.hand_detection_queue),
            ocr_detector = WildOcrDecetor(self.inference_event, self.done_barrier, self.frame_shared_array, self.wild_ocr_queue),
        )

        for p in self.subprocesses.values():
            p.start()

    def run(self):
        for frame in self.stream:
            self.share_frame(frame)
            
            hand_detection_results, drug_detection_results, wild_ocr_results = self.parallel_inference()
            
            self.catch_checker.observe(hand_detection_results, drug_detection_results)
            check_results = self.catch_checker.check()

            self.export_results(frame, check_results, hand_detection_results, drug_detection_results, self.catch_checker.hand_tracker.tracked_objects, self.catch_checker.medicine_tracker.tracked_objects)
    
    def share_frame(self, frame: np.ndarray) -> None:
        np.copyto(np.frombuffer(self.frame_shared_array.get_obj(), dtype=np.uint8), frame.flatten())

    def parallel_inference(self):
        # start parallel inference
        self.inference_event.set()
        self.done_barrier.wait()
        
        # finish parallel inference
        self.inference_event.clear()
        self.done_barrier.reset()
        
        # get inference results
        hand_detection_results = self.hand_detection_queue.get()
        drug_detection_results = self.drug_detection_queue.get()
        wild_ocr_results = self.wild_ocr_queue.get()
        
        return hand_detection_results, drug_detection_results, wild_ocr_results

    def export_results(self, frame, check_results, hand_detection_results, drug_detection_results, hand_tracked, drug_tracked) -> None:
        self.plot_results(frame, check_results, hand_detection_results, drug_detection_results, hand_tracked, drug_tracked)        
        print("---------------------------------------------------------------------")
        print(check_results)
        print(hand_detection_results)
        print(drug_detection_results)
        print(hand_tracked)
        print(drug_tracked)

    def plot_results(self, frame, check_results, hand_detection_results, drug_detection_results, hand_tracked, drug_tracked):
        image = deepcopy(frame)
        
        for hand in hand_detection_results:
            plot_xywh(image, np.array(hand['bbox'], dtype=int), category=hand['category_id'])
            
        for drug in drug_detection_results:
            plot_xywh(image, np.array(drug['bbox'], dtype=int), category=drug['category_id'])
            
        for object_catched in check_results:
            plot_xywh(image, np.array(object_catched.get_latest_valid_node().bbox, dtype=int), category=object_catched.category_id, color=(0, 0, 255))
            
        cv2.imshow('img', image)
        cv2.waitKey(1)
        
if __name__ == '__main__':
    test1 = MainProcess()
    test1.run()