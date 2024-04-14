
import multiprocessing
import time
import cv2
import numpy as np
from typing import Union, List, Dict, Any
from modules.yoloc_dector import YolovDector
from modules.hand_detector import HandDetector
from modules.catch_checker import CatchChecker
from modules.camera_processor import CameraProcessor


class MainProcess:
    def __init__(self) -> None:
        # self.cam_stream = cv2.VideoCapture("/home/portable-00/VisionCopilot/test/20240322_151423.mp4")
        self.running_process: multiprocessing.Process = None
        self.cam_stream = CameraProcessor()
        self.yoloc_decctor = YolovDector()
        self.hand_detector = HandDetector()
        self.catch_checker = CatchChecker()
        
    def referer(self):
        for frame_id, frame in enumerate(self.video_flow):
            prescription , res_array = self.yoloc_decctor._scan_prescription(frame)
            if res_array[0] in prescription and res_array[1] in prescription:
                if self.running_process is not None and self.running_process.is_alive():
                    self.running_process.terminate()
                self.running_process = self._run_real_scene(prescription)
                time.sleep(2)
            
    def _run_real_scene(self,prescription):
        p1 = multiprocessing.Process(target=self.process_stream, args=(prescription))
        p1.daemon = True
        p1.start()
        return p1        
    
    def process_stream(self):
        while True:
            frame: np.ndarray = self.capture_frame()
            prescription , res_array = self.yoloc_decctor.__scan_prescription(frame)
            print(prescription)
            print(res_array)
            if res_array[0][0] in prescription and res_array[1][0] in prescription:
                break
            
        while True:
            frame: np.ndarray = self.capture_frame()

            # print(rf"-------------------------------------------- Frame {frame_id} --------------------------------------------")
            medicines_detections = self.detect_medicines(frame)

            if medicines_detections is None or medicines_detections[0] == 'nomatch':
                continue

            # drug_match: bool = self.yoloc_decctor.drug_match(medicines_detections[0][0], prescription)

            hands_detections = self.detect_hands(frame)
            print(rf"Hand detection result: ")
            for i, hand in enumerate(hands_detections):
                print(rf"{i}: Bbox {np.array(hand['bbox'], dtype=int)}")
                
            offer_value = [{"category_id":int(medicines_detections[0][0]),"bbox":medicines_detections[1][0]}]
            print(rf"Medicines detection result: ")
            for i, medicine in enumerate(offer_value):
                print(rf"{i}: Bbox {np.array(medicine['bbox'], dtype=int)}")

            self.track_objects(hands_detections, offer_value)
            check_results = self.catch_recognition()

            print('')
            print(rf"Catch check result: ")
            for i, check_result in enumerate(check_results):
                print(rf"{i}: {check_result.category_id}")
                if self.drug_match(check_result.category_id,prescription):
                    self.cam_stream.send_wrong()
                
    
    def capture_frame(self) -> np.ndarray:
        while True:
            valid, frame = self.cam_stream.achieve_image()
            if valid:
                return frame

    def _scan_prescription(self, frame: np.ndarray) -> List[Any]:
        return self.yoloc_decctor.scan_prescription(frame)

    def detect_medicines(self, frame: np.ndarray) -> List[Dict]:
        return self.yoloc_decctor.detect_medicines(frame)

    def detect_hands(self, frame: np.ndarray) -> List[Dict]:
        return self.hand_detector.detect(frame)

    def track_objects(self, hands_detections: List[Dict], medicines_detections: List[Dict]) -> None:
        self.catch_checker.observe(hands_detections, medicines_detections)

    def catch_recognition(self) -> List[Dict]:
        return self.catch_checker.check()

    def drug_match(self, medicine_cls, prescription):
        return self.yoloc_decctor.drug_match(self, medicine_cls, prescription)


if __name__ == '__main__':
    test1 = MainProcess()
    test1.referer()
