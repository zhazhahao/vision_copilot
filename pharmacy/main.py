import numpy as np
from typing import Union, List, Dict, Any
from config import *
from utils.ocr_infer.predict_system import *


class MainProcess:
    def __init__(self) -> None:
        self.camera = CameraProcessor()
        self.camera.start()

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
            
            self.catch_recognition()

    def capture_frame(self) -> np.ndarray:
        while True:
          bools,mat = self.camera.achieve_image()
          if bools:
              
              return mat
              
              
    def scan_prescription(self, frame: np.ndarray) -> List[Any]:
        prescription_list = procession(frame,text_sys,data_lists,"prescription")


    def detect_medicines(self, frame: np.ndarray) -> List[Dict]:
        res = yolo_and_ocr_0(frame)
        

    def detect_hands(self, frame: np.ndarray) -> List[Dict]:
        ...
    

    def track_objects(self, medicines_detections: List[Dict], hands_detections: List[Dict]) -> None:
        ...
    
    
    def catch_recognition(self) -> List[Dict]:
        ...
        
        
    def drug_match(self, medicine_cls, prescription):
        med_name = get_drug_by_index(medicine_cls,data)
        if med_name["ҩƷ��"] in prescription:
            return True
        else:
            return False
        
test1 = MainProcess()
test1.run()