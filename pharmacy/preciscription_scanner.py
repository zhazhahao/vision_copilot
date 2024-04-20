from collections import Counter
import os
import time
import cv2
import numpy as np
import utils.ocr_infer.pytorchocr_utility as utility
from utils.ocr_infer.predict_system import TextSystem
from utils.ocr_infer.ocr_processor import procession
from utils.ocr_infer.load_data_list import load_txt
from qinglang.utils.utils import ClassDict

def get_numeric_part(filename):
    numeric_part = ''.join(filter(str.isdigit, filename))
    return int(numeric_part) if numeric_part else 0

class PreScriptionRecursiveObject:
    def __init__(self) -> None:
        self.from_index = 0
        self.to_index = 0 
        self.times = 0
        self.recursive_epoch = 3
        self.recursive_obj = []
        self.static_obj = []
    def update(self, list,times) -> None:
        self.recursive_obj = self._merge_drug_lists(self.recursive_obj,list, times)
        
    def achieve(self):
        return self.static_obj.extend(self.recursive_obj)
    
    def _merge_drug_lists(self, list1, list2, times):
        checked_list_time = Counter()
        merged_list = []
        reserve_list = []
        i,j = 0,0
        first_there = True
        if len(list1) == 0:
            return list2
        if len(list1) > 1 and len(list2) == 1:
            return list1
        while i < len(list1) and j < len(list2):
            if list1[i] == list2[j]:
                self.from_index = i if first_there else self.from_index
                merged_list.append(list1[i]) 
                first_there = False
                i += 1
                j += 1
            elif list2[j] in list1 and list1.index(list2[j]) >= i:
                # if list1.index(list2[j]) == i + 1:
                #     merged_list.extend(reserve_list)
                reserve_list = []
                merged_list.append(list1[i])
                i += 1
            else:
                self.from_index = i if first_there else self.from_index
                first_there = False
                reserve_list.append(list2[j])
                j += 1
        merged_list.extend(list1[i:])
        merged_list.extend(list2[j:])
        self.to_index = merged_list.__len__()
        if len(reserve_list) > len(list2) - len(reserve_list):
            print(times)
            self.static_obj.append([times,merged_list])
            merged_list = []
        merged_list.extend(reserve_list)
        # for item in reversed(merged_list):
        #     result_list.insert(0, item) if item not in result_list else None
        return merged_list

class FrameMaxMatchingCollections(ClassDict):
    def __init__(self, *args, **kwargs) -> None:
        self.result_counter = {}
        
    def update(self ,frame, max_candicated, times):
        for result in max_candicated[1]:
            self.result_counter[result] = ClassDict(
                    tickles = times,
                    max_candicated = max_candicated,
                    res_frame = frame,
                    counts = 1 if result not in self.result_counter.keys() else self.result_counter[result].counts + 1
                ) if result not in self.result_counter or max_candicated[1].__len__() > self.result_counter[result].max_candicated[1].__len__() else ClassDict(
                    tickles = self.result_counter[result].tickles,
                    max_candicated = self.result_counter[result].max_candicated,
                    res_frame = self.result_counter[result].res_frame,
                    counts = self.result_counter[result].counts + 1
                )
    def values(self):
        return {elements.tickles: ClassDict(frame=elements.res_frame, max_candicated=elements.max_candicated) 
                              for i, elements in self.result_counter.items()}
class OCRProcess:
    def __init__(self) -> None:
        self.test = True
        self.max_opportunity = 10
        self.enlarge_bbox_ratio = 0.2
        self.max_trigger_slam = 5
        self.data_lists = load_txt("/home/portable-00/VisionCopilot/pharmacy/database/medicine_names.txt")
        self.text_sys = TextSystem(utility.parse_args())
        self.candiancate = PreScriptionRecursiveObject()
        self.frame_collections = FrameMaxMatchingCollections()
        
    def scan_prescription(self):
        end_trigger_times = 0
        times = 0
        status_dict = {}
        res_counter = []
        counter = Counter()
        for filename in sorted(os.listdir(r"/home/portable-00/data/images"),key=get_numeric_part):
            times += 1
            frame = cv2.imread(r"/home/portable-00/data/images/"+filename)
            (dt_box_res,prescription,trigger) = procession(frame,self.text_sys,self.data_lists,"prescription")
            end_trigger_times += 1 if trigger else 0
            counter.update(prescription)
            res_frame , res_counter= frame , [dt_box_res,prescription]
            height,width = self.getavgSize(res_counter[0])
            for res in res_counter[0]:
                res_frame = cv2.rectangle(res_frame, tuple(res[0].astype("int")),tuple(res[2].astype("int")),color=(0, 255, 0),thickness=-1)
            conter_len = 0 
            
            for i in range(1, len(res_counter[0])):
                if (res_counter[0][i][3][1] - res_counter[0][i-1][0][1]) >= height * 1.5:
                    fix_height = res_counter[0][i-1][0][1]
                    first_set = True
                    while fix_height < res_counter[0][i][0][1]:
                        selected_height = min(int(res_counter[0][i - 1][2][1]),int(res_counter[0][i - 1][3][1])) if first_set else int(selected_height + height)
                        selected_width  = min(int(res_counter[0][i - 1][3][0]),int(res_counter[0][i][3][0]))
                        first_set = False
                        if selected_height + int(height) > res_frame.shape[0]:
                            selected_height = res_frame.shape[0]
                        rec_res = procession(res_frame[selected_height + int(height * 0.5):selected_height + int(height * 1.5),
                                                   selected_width:selected_width + int(width * 1.5)]
                                         ,self.text_sys,data_lists=self.data_lists,options="Single")
                        if self.test:
                            cv2.imwrite("refe/"+str(filename)+"_"+str(conter_len)+".jpg",res_frame[selected_height + int(height * 0.5):selected_height + int(height * 1.5),
                                                       selected_width:selected_width + int(width * 1.5)]
                                             )
                        status_dict[rec_res] = "Update"
                        fix_height += height
                        if rec_res == None:
                            fix_height += height
                            continue
                        conter_len += 1
                        counter.update([rec_res])
                        res_counter[1].insert(i - 1 + conter_len,rec_res)
            
            self.frame_collections.update(frame,max_candicated=[dt_box_res,prescription],times=times)
            if self.test:
                cv2.imwrite(str(times)+".png",res_frame)
            if res_counter[1].__len__() > 0:
                self.candiancate.update(res_counter[1],times)
            if end_trigger_times == self.max_opportunity:
                print(times)
                break
        
        tools = self.frame_collections.values()
        res_con = [answer if not(answer in counter.keys() and counter[answer] == 1 ) or answer in status_dict.keys() else None for answer in self.candiancate.recursive_obj]
        print(self.frame_collections.result_counter)
        print(status_dict)
        print(self.candiancate.recursive_obj)
        
    def getavgSize(dt_boxes):
        if dt_boxes is not None and len(dt_boxes) != 0:
            rects = np.array(dt_boxes)
            heights = rects[:, 3, 1] - rects[:, 0, 1] + rects[:, 2, 1] - rects[:, 1, 1]
            widths = rects[:, 2, 0] - rects[:, 0, 0] - rects[:, 3, 0] + rects[:, 1, 0]
            return heights.mean()/2 , widths.mean()/2
        else:
            return 0, 0
        
test = OCRProcess()
test.scan_prescription()