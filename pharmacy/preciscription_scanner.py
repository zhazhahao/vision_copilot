from collections import Counter
import os
import time
import cv2
import utils.ocr_infer.pytorchocr_utility as utility
from utils.ocr_infer.predict_system import TextSystem
from utils.ocr_infer.ocr_processor import procession
from utils.ocr_infer.load_data_list import load_txt
def get_numeric_part(filename):
    # 提取文件名中的数字部分
    numeric_part = ''.join(filter(str.isdigit, filename))
    return int(numeric_part) if numeric_part else 0
def getavgSize(dt_boxes):
    if dt_boxes is not None:
        rects =[dt_box for dt_box in dt_boxes]
        avgWidth = sum([rect[2][0]-rect[0][0]-rect[3][0]+rect[1][0] for rect in rects])/dt_boxes.__len__()/2
        avgHeight = sum([rect[3][1]-rect[0][1]+rect[2][1]-rect[1][1] for rect in rects])/dt_boxes.__len__()/2
        return avgHeight,avgWidth
    else:
        return 0,0
def merge_drug_lists(list1, list2):
        merged_list = []
        reserve_list = []
        i,j = 0,0
        if len(list1) > 1 and len(list2) == 1:
            return list1
        # 合并两个列表
        while i < len(list1) and j < len(list2):
            # 如果当前元素相等，添加到合并列表中
            if list1[i] == list2[j]:
                merged_list.append(list1[i]) if list1[i] not in merged_list else None
                i += 1
                j += 1
            elif list2[j] in list1:
                if list1.index(list2[j]) > i:
                    merged_list.extend(reserve_list)
                    reserve_list = []
                    merged_list.append(list1[i])
                    i += 1
                else:
                    merged_list.append(list2[j])
                    j += 1
            else:
                reserve_list.append(list2[j])
                j += 1
        merged_list.extend(list1[i:])
        merged_list.extend(list2[j:])
        result_list = []
        for item in reversed(merged_list):
            result_list.insert(0, item) if item not in result_list else None
        return result_list
class OCRProcess:
    def __init__(self) -> None:
        # self.max_opportunity = 10
        self.max_trigger_slam = 5
        self.data_lists = load_txt("/home/portable-00/VisionCopilot/pharmacy/database/medicine_names.txt")
        self.text_sys = TextSystem(utility.parse_args())
        self.candiancate = []
    
    def scan_prescription(self):
        end_trigger_times = 0
        max_candicated = 0
        res_counter = []
        result_counter = Counter()
        for filename in sorted(os.listdir(r"/home/portable-00/VisionCopilot/pharmacy/images"),key=get_numeric_part):
            frame = cv2.imread(r"/home/portable-00/VisionCopilot/pharmacy/images/"+filename)
            (dt_box_res,prescription,trigger) = procession(frame,self.text_sys,self.data_lists,"prescription")
            end_trigger_times += 1 if trigger else 0
            if max_candicated < prescription.__len__():
                max_candicated = prescription.__len__()
                res_counter = [dt_box_res,prescription]
                res_frame = frame
            print(prescription)
            for result in prescription:
                result_counter[result] += 1
            self.candiancate = merge_drug_lists(self.candiancate,prescription) # Waiting For implemention
        for res in res_counter[0]:
            res_frame = cv2.rectangle(res_frame, tuple(res[0].astype("int")),tuple(res[2].astype("int")),color=(0, 255, 0),thickness=-1)
        cv2.imshow("real_img",res_frame)
        height,width = getavgSize(res_counter[0])
        conter_len = 0
        for i in range(1, len(res_counter[0])):
            if (res_counter[0][i][0][1] - res_counter[0][i-1][0][1]) >= height * 1.5:
                res = procession(res_frame[int(res_counter[0][i-1][3][1]):int(res_counter[0][i][2][1]),
                                           int(res_counter[0][i][3][0]):int(res_counter[0][i-1][2][0])]
                                 ,self.text_sys,data_lists=self.data_lists,options="Single")
                res_counter[1].insert(i + conter_len,res)
                conter_len += 1
        cv2.waitKey(0)
        
        print(res_counter[1])
        print(result_counter)
        print(self.candiancate)
test = OCRProcess()
test.scan_prescription()