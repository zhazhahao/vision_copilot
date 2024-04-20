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
from qinglang.data_structure.video.video_base import VideoFlow
import re

def remove_last_digit_and_after(string):
    # 使用正则表达式匹配字符串中最后一个数字及其后面的字符，并替换为空字符串
    return re.sub(r'\d+[^0-9]*$', '', string)

# 示例用法
string1 = "abc123def456aaa"
string2 = "xyz123444aaaaa"
print(remove_last_digit_and_after(string1))  # 输出 "abc123def"
print(remove_last_digit_and_after(string2))  # 输出 "xyz"
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
            if res_array[0][0] in prescription or res_array[1][0] in prescription:
                for result in prescription:
                    result_counter[result] += 1
                opportunity += 1
                if opportunity >= self.max_opportunity:
                    break
        prescription = [result for result, count in result_counter.items() if count >= 5 and result != res_array[0][0] and result != res_array[1][0]]
        check_list = []
        while True:
            frame: np.ndarray = self.capture_frame()
            hands_detections = self.detect_hands(frame)
            hand_output = ''
            medicine_output = ''
            for i, hand in enumerate(hands_detections):
                handresult = f"{i}: Bbox {np.array(hand['bbox'], dtype=int)}"
                hand_output += "\n"
                hand_output += handresult
                
            medicines_detections = self.detect_medicines(frame)

            if medicines_detections is not None and medicines_detections[0] != 'nomatch':
                medicines_detections = [{"category_id":int(medicines_detections[0][i]),"bbox":medicines_detections[1][i]} for i in range(len(medicines_detections[0]))]
                for i, medicine in enumerate(medicines_detections):
                    medresult = f"{i}: Category {medicine_lookup_table[medicine['category_id']]}"
                    medicine_output += "\n"
                    medicine_output += medresult
                    
                self.track_objects(hands_detections, medicines_detections)
                check_results = self.catch_recognition()
                for i, check_result in enumerate(check_results):
                    # print(rf"{i}: {check_result.category_id}")

                    check_list.append(check_result.category_id) if self.medicine_match(check_result.category_id, prescription) and check_result.category_id not in check_list else self.cam_stream.send_wrong()
                if check_list.__len__() == prescription.__len__():
                    print("All Done")
                    break  
                # medicines_detections = [{"category_id":int(medicines_detections[0][i]),"bbox":medicines_detections[1][i]} for i in range(len(medicines_detections[0]))]
            
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
    

            

# 在主程序中调用 static_refresh 方法
if __name__ == '__main__':
    test = MainProcess()
    prescription = ["曲前列尼尔注射液", "速碧林(那屈肝素钙注射液)", "依诺肝素钠注射液", "注射用青霉素钠", "注射用头孢唑林钠"]
    hand_output = "0: Bbox [1, 1, 1, 1]"
    medicine_output = "123: Category 一二三"
    
    def merge_drug_lists(list1, list2):
        merged_list = []
        i = 0
        j = 0

        # 合并两个列表
        while i < len(list1) and j < len(list2):
            # 如果当前元素相等，添加到合并列表中
            if list1[i] == list2[j]:
                if list1[i] not in merged_list:
                    merged_list.append(list1[i])
                i += 1
                j += 1
            # 否则按顺序添加较小的元素
            elif list1[i] < list2[j]:
                if list1[i] not in merged_list:
                    merged_list.append(list1[i])
                i += 1
            else:
                if list2[j] not in merged_list:
                    merged_list.append(list2[j])
                j += 1

        # 将剩余的元素添加到合并列表中
        while i < len(list1):
            if list1[i] not in merged_list:
                merged_list.append(list1[i])
            i += 1

        while j < len(list2):
            if list2[j] not in merged_list:
                merged_list.append(list2[j])
            j += 1

        return merged_list

# 示例
list1 = ['维生素C注射液', '肝素钠注射液', '胰岛素注射液']
list2 = ['左氧氟沙星氯化钠注射液', '维生素C注射液', '氯化钙注射液', '肝素钠注射液', '怡万之(注射用厄他培南)', '胰岛素注射液']
result = merge_drug_lists(list1, list2)
print(result)
