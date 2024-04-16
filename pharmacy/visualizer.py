import cv2
import numpy as np
from typing import Union, List, Dict, Any
from modules.yoloc_dector import YolovDector
from modules.hand_detector import HandDetector
from modules.catch_checker import CatchChecker
from modules.camera_processor import CameraProcessor
from qinglang.data_structure.video.video_base import VideoFlow
from qinglang.dataset.utils.utils import plot_xywh, xywh2xyxy, plot_xyxy, centerwh2xywh
from pprint import pprint


class MainProcess:
    def __init__(self) -> None:
        self.video_flow = VideoFlow("/home/portable-00/VisionCopilot/pharmacy/20240313_160556/20240313_160556.mp4")
        self.yoloc_decctor = YolovDector()
        self.hand_detector = HandDetector()
        self.catch_checker = CatchChecker()

    def run(self):
        prescription = ["曲前列尼尔注射液", "速碧林(那屈肝素钙注射液)", "依诺肝素钠注射液","注射用青霉素钠","注射用头孢唑林钠"]
        check_list = []
        for frame_id, frame in enumerate(self.video_flow):
            print(rf"-------------------------------------------- Frame {frame_id} --------------------------------------------")

            hands_detections = self.detect_hands(frame)
            print(rf"Hand detection result: ")
            for i, hand in enumerate(hands_detections):
                print(rf"{i}: Bbox {np.array(hand['bbox'], dtype=int)}")

            for hand in hands_detections:
                plot_xywh(frame, np.array(hand['bbox'], dtype=int))

            medicines_detections = self.detect_medicines(frame)

            if medicines_detections is not None and medicines_detections[0] != 'nomatch':

                for medicine in medicines_detections[1]:
                    plot_xywh(frame, centerwh2xywh(np.array(medicine, dtype=int)), color=(0, 0, 255))

                medicines_detections = [{"category_id":int(medicines_detections[0][i]),"bbox":medicines_detections[1][i]} for i in range(len(medicines_detections[0]))]
                print(rf"Medicines detection result: ")
                for i, medicine in enumerate(medicines_detections):
                    print(rf"{i}: Bbox {np.array(medicine['bbox'], dtype=int)}")
                
                self.track_objects(hands_detections, medicines_detections)
                check_results = self.catch_recognition()

                print('')
                print(rf"Catch check result: ")
                for i, check_result in enumerate(check_results):
                    print(rf"{i}: {check_result.category_id}")
                    check_list.append(check_result.category_id) if self.medicine_match(check_result.category_id, prescription) and check_result.category_id not in check_list else print(11111111111111111)
                if check_list.__len__() == prescription.__len__():
                    print("All Done")
                    break      
            frame = cv2.resize(frame,(800,400))
            cv2.imshow("vis", frame)
            cv2.waitKey(1)

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
        
    def medicine_match(self, medicine_cls, prescription):
        return self.yoloc_decctor.drug_match(medicine_cls, prescription)
        
if __name__ == '__main__':
    test1 = MainProcess()
    test1.run()