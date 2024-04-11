import numpy as np
from typing import Union, List, Dict, Any

from modules.yoloc_dector import YolovDector
from modules.hand_detector import HandDetector
from modules.catch_checker import CatchChecker
from modules.camera_processor import CameraProcessor


class MainProcess:
    def __init__(self) -> None:
        self.camera = CameraProcessor()
        self.yoloc_decctor = YolovDector()
        self.hand_detector = HandDetector()
        self.catch_checker = CatchChecker()

    def run(self):
        while True:
            frame: np.ndarray = self.capture_frame()
            prescription: List[Any] = self.scan_prescription(frame)
            if "����ҩҩ������" in prescription and "�ϼ�" in prescription:
                break
        
        while True:
            frame: np.ndarray = self.capture_frame()

            medicines_detections = self.detect_medicines(frame)
            
            drug_match: bool = drug_match(self, medicines_detections[0][0], prescription)

            hands_detections = self.detect_hands(frame)

            self.track_objects(medicines_detections, hands_detections)
            
            print(self.catch_recognition())

    def capture_frame(self) -> np.ndarray:
        while True:
          bools,mat = self.camera.achieve_image()
          if bools:
              
              return mat
              
              
    def scan_prescription(self, frame: np.ndarray) -> List[Any]:
        return self.yoloc_decctor.scan_prescription(frame)


    def detect_medicines(self, frame: np.ndarray) -> List[Dict]:
        return self.detect_medicines(frame)
        

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
    test1.run()