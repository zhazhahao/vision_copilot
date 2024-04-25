import os
import cv2
import numpy as np
import torch.multiprocessing as multiprocessing
from copy import deepcopy
from datetime import datetime
from typing import List
from modules.cameras import VirtualCamera, DRIFTX3
from modules.catch_checker import CatchChecker
from modules.drug_detector_process import DrugDetectorProcess
from modules.hand_detector_process import HandDetectorProcess
from modules.ocr_process import OCRProcess
from utils.utils import MedicineDatabase
from qinglang.utils.utils import ClassDict, Config, most_common
from qinglang.dataset.utils.utils import plot_xywh


# medicine_database = MedicineDatabase()

# with open("prescription.txt", "r") as f:
#     med_list = [line.strip() for line in f.readlines()]

# med_ids = [m['Category ID'] for med in med_list for m in medicine_database.find(med)]


class MainProcess:
    def __init__(self) -> None:
        multiprocessing.set_start_method('spawn')
 
        self.config = Config("configs/main.yaml")
        self.source = Config("configs/source.yaml")
        
        self.status = ClassDict(
            current_task = 'scan'
        )
        
        self.init_work_dir()
        self.init_shared_variables()
        self.init_subprocess()

        # self.stream = DRIFTX3()
        self.stream = VirtualCamera(self.source.virtual_camera_source)
        self.medicine_database = MedicineDatabase()
        self.catch_checker = CatchChecker()

    def init_work_dir(self) -> None:
        self.work_dir = f"work_dirs/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        os.makedirs(self.work_dir, exist_ok=True)

        if self.config.export_results_images:
            self.image_dir = os.path.join(self.work_dir, 'images')
            os.makedirs(self.image_dir, exist_ok=True)

    def init_shared_variables(self) -> None:
        self.frame_shared_array = multiprocessing.Array('B', 1920 * 1080 * 3)
        self.inference_event = multiprocessing.Event()
        self.done_barrier = multiprocessing.Barrier(4)
        self.hand_detection_queue = multiprocessing.Queue()
        self.drug_detection_queue = multiprocessing.Queue()
        self.ocr_queue = multiprocessing.Queue()

    def init_subprocess(self) -> None:
        self.subprocesses = ClassDict(
            drug_detector = DrugDetectorProcess(self.inference_event, self.done_barrier, self.frame_shared_array, self.drug_detection_queue),
            hand_detector = HandDetectorProcess(self.inference_event, self.done_barrier, self.frame_shared_array, self.hand_detection_queue),
            ocr = OCRProcess(self.inference_event, self.done_barrier, self.frame_shared_array, self.ocr_queue),
        )

        for p in self.subprocesses.values():
            p.start()

    def run(self) -> None:
        for frame in self.stream:
            self.share_frame(frame)
            
            if self.status.task == 'scan':
                self.prescription = self.get_prescription()
                self.drugs_caught = []
                
                self.status.task == 'check'
            
            if self.status.task == 'check':
                hand_detection_results, drug_detection_results, ocr_results = self.parallel_inference()
                drug_detection_results = self.fiter_drug_detection_results(ocr_results, drug_detection_results)

                self.update_observation(hand_detection_results, drug_detection_results)
                check_results = self.check()

                self.export_results(frame, check_results, hand_detection_results, drug_detection_results, self.catch_checker.hand_tracker.tracked_objects, self.catch_checker.medicine_tracker.tracked_objects)

    def share_frame(self, frame: np.ndarray) -> None:
        np.copyto(np.frombuffer(self.frame_shared_array.get_obj(), dtype=np.uint8), frame.flatten())

    def get_prescription(self) -> List[int]:
        return self.subprocesses.ocr.scan_prescription(self.stream)

    def parallel_inference(self) -> None:
        # start parallel inference
        self.inference_event.set()
        self.done_barrier.wait()
        
        # finish parallel inference
        self.inference_event.clear()
        self.done_barrier.reset()
        
        # get inference results
        hand_detection_results = self.hand_detection_queue.get()
        drug_detection_results = self.drug_detection_queue.get()
        ocr_results = self.ocr_queue.get()
        
        return hand_detection_results, drug_detection_results, ocr_results
    
    def fiter_drug_detection_results(self, ocr_results, drug_detection_results):
        if ocr_results:
            location = most_common([self.medicine_database[medicine['category_id']].get("Shelf") for medicine in ocr_results])
            drug_detection_results = [drug for drug in drug_detection_results if self.medicine_database[drug['category_id']].get("Shelf") == location]
        return drug_detection_results

    def update_observation(self, hand_detection_results, drug_detection_results):
        self.catch_checker.observe(hand_detection_results, drug_detection_results)
    
    def check(self):
        check_results = self.catch_checker.check()
        
        for object_catched in check_results:
            if object_catched.category_id in self.prescription and object_catched.category_id not in self.drugs_caught:
                self.drugs_caught.append(object_catched.category_id)
            else:
                self.stream.beep()
        
        return check_results
                
    def export_results(self, frame, check_results, hand_detection_results, drug_detection_results, hand_tracked, drug_tracked) -> None:
        if self.config.export_results_images:
            self.plot_results(frame, check_results, hand_detection_results, drug_detection_results)

        print("---------------------------------------------------------------------")
        print(check_results)
        print(hand_detection_results)
        print(drug_detection_results)
        print(hand_tracked)
        print(drug_tracked)

    def plot_results(self, frame, check_results, hand_detection_results, drug_detection_results):
        image = deepcopy(frame)
        
        for hand in hand_detection_results:
            plot_xywh(image, np.array(hand['bbox'], dtype=int), category='手')
            
        for drug in drug_detection_results:
            plot_xywh(image, np.array(drug['bbox'], dtype=int), category=self.medicine_database[drug['category_id']]['Name'])
            
        for object_catched in check_results:
            plot_xywh(image, np.array(object_catched.get_latest_valid_node().bbox, dtype=int), color=(0, 0, 255))
        
        cv2.imwrite(os.path.join(self.image_dir, datetime.now().strftime("%Y%m%d-%H%M%S-%f") + '.jpg'), image)

if __name__ == '__main__':
    test1 = MainProcess()
    test1.run()
