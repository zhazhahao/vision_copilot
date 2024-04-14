import multiprocessing
import time
import cv2
import numpy as np
from typing import Union, List, Dict, Any
from modules.yoloc_dector import YolovDector
from modules.hand_detector import HandDetector
from modules.catch_checker import CatchChecker
from modules.camera_processor import CameraProcessor
from qinglang.data_structure.video.video_base import VideoFlow
from pprint import pprint


class MainProcess:
    def __init__(self) -> None:
        self.video_flow = VideoFlow("/home/portable-00/VisionCopilot/test/test_1.mp4")
        self.yoloc_decctor = YolovDector()
        self.hand_detector = HandDetector()
        self.catch_checker = CatchChecker()
    def referer(self):
        for frame_id, frame in enumerate(self.video_flow):
            prescription , res_array = self.yoloc_decctor.scan_prescription(frame)
            if res_array[0] in prescription and res_array[1] in prescription:
                self.running_process = self.run_real_scene(prescription)
                time.sleep(2)
                break
            
    def run_real_scene(self,prescription):
        p1 = multiprocessing.Process(target=self.process_stream, args=(prescription))
        p1.daemon = True
        p1.start()
        return p1        
    
    def run_test(self):
        prescription = ["曲前列尼尔注射液", "速碧林(那屈肝素钙注射液)", "依诺肝素钠注射液"]
        # 使用多进程处理视频帧
        p1 = multiprocessing.Process(target=self.process_stream, args=(prescription))
        p1.start()
        return p1
    
    def process_stream(self,frame,prescription):
        
        for frame_id, frame in enumerate(self.video_flow):
            print(rf"-------------------------------------------- Frame {frame_id} --------------------------------------------")
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
              
    def scan_prescription(self, frame: np.ndarray) -> List[Any]:
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
    test1.run_test()