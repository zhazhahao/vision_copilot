import os
import time
import cv2
import sys
import numpy as np
from collections import Counter
from typing import Union, List, Dict, Any
from modules.yoloc_dector import YolovDector
from modules.hand_detector import HandDetector
from modules.catch_checker import CatchChecker
from modules.camera_processor import CameraProcessor
from qinglang.dataset.utils.utils import plot_xywh,centerwh2xywh
from qinglang.data_structure.video.video_base import VideoFLow

medicine_lookup_table = []
with open("/home/portable-00/VisionCopilot/pharmacy/database/medicine_names.txt", 'r') as f:
    for line in f:
        medicine_lookup_table.append(line)

class MainProcess:
    def __init__(self) -> None:
        self.video_flow = VideoFlow("/home/portable-00/VisionCopilot/test/test_1.mp4")
        self.max_opportunity = 10
        self.yoloc_decctor = YolovDector()
        self.hand_detector = HandDetector()
        self.catch_checker = CatchChecker()

    def run(self):
        opportunity = 0
        result_counter = Counter()
        for frame in self.video_flow():
            prescription, res_array = self.yoloc_decctor.scan_prescription(frame)
            
            print(prescription)
            if res_array[0][0] in prescription or res_array[1][0] in prescription:
                # print(opportunity)
                for result in prescription:
                    result_counter[result] += 1
                opportunity += 1
                if opportunity >= self.max_opportunity:
                    break
        prescription = [result for result, count in result_counter.items() if count >= 5 and result != res_array[0][0] and result != res_array[1][0]]
        print("prescription:",prescription)
        check_list = []
        while True:
            sys.stdout.write("\033[F")

            # self.clear_console()
            frame: np.ndarray = self.capture_frame()
            hands_detections = self.detect_hands(frame)
            print(rf"Hand detection result: ")
            print(rf"----------------------------------------------------------------------------------------")
            for i, hand in enumerate(hands_detections):
                print(rf"{i}: Bbox {np.array(hand['bbox'], dtype=int)}")
            print(rf"----------------------------------------------------------------------------------------")

            medicines_detections = self.detect_medicines(frame)

            if medicines_detections is not None and medicines_detections[0] != 'nomatch':

                for medicine in medicines_detections[1]:
                    plot_xywh(frame, centerwh2xywh(np.array(medicine, dtype=int)), color=(0, 0, 255))

                medicines_detections = [{"category_id":int(medicines_detections[0][i]),"bbox":medicines_detections[1][i]} for i in range(len(medicines_detections[0]))]
                print(rf"Medicines detection result: ")
                print(rf"----------------------------------------------------------------------------------------")
                for i, medicine in enumerate(medicines_detections):
                    print(rf"{i}: Category {medicine_lookup_table[medicine['category_id']]}")
                print(rf"----------------------------------------------------------------------------------------")
                
                self.track_objects(hands_detections, medicines_detections)
                check_results = self.catch_recognition()

                print('')
                print(rf"Catch check result: ")
                for i, check_result in enumerate(check_results):
                    # print(rf"{i}: {check_result.category_id}")

                    check_list.append(check_result.category_id) if self.medicine_match(check_result.category_id, prescription) and check_result.category_id not in check_list else self.cam_stream.send_wrong()
                print(rf"----------------------------------------------------------------------------------------")
                print("check_list",check_list)
                print("Prescription",prescription)
                print(rf"----------------------------------------------------------------------------------------")
                if check_list.__len__() == prescription.__len__():
                    print("All Done")
                    break  
            else:
                # medicines_detections = [{"category_id":int(medicines_detections[0][i]),"bbox":medicines_detections[1][i]} for i in range(len(medicines_detections[0]))]
                print(rf"Medicines detection result: ")
                print(rf"----------------------------------------------------------------------------------------")
                print(rf"None")
                print(rf"----------------------------------------------------------------------------------------")
                
                print(rf'')
                print(rf"Catch check result: ")
                print(rf"----------------------------------------------------------------------------------------")
                print(rf'None')
                print(rf"check_list", check_list)
                print(rf"Prescription", prescription)
                print(rf"----------------------------------------------------------------------------------------")            
            # cv2.imshow("vis", frame)
            # cv2.waitKey(10)

    def capture_frame(self) -> np.ndarray:
        while True:
            valid, frame = self.cam_stream.achieve_image()
            if valid:
                return frame
            else:
                time.sleep(0.01)

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
    
    def clear_console(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        
if __name__ == '__main__':
    test1 = MainProcess()
    test1.run()
