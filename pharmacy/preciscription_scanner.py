from collections import Counter
import os
import time
import cv2
import utils.ocr_infer.pytorchocr_utility as utility
from utils.ocr_infer.predict_system import TextSystem
from utils.ocr_infer.ocr_processor import procession
from utils.ocr_infer.load_data_list import load_txt
from qinglang.utils.utils import ClassDict

def get_numeric_part(filename):
    numeric_part = ''.join(filter(str.isdigit, filename))
    return int(numeric_part) if numeric_part else 0

def getavgSize(dt_boxes):
    if dt_boxes is not None:
        rects =[dt_box for dt_box in dt_boxes]
        return sum([rect[3][1]-rect[0][1]+rect[2][1]-rect[1][1] for rect in rects])/dt_boxes.__len__()/2,sum([rect[2][0]-rect[0][0]-rect[3][0]+rect[1][0] for rect in rects])/dt_boxes.__len__()/2
    else:
        return 0,0
    
class OCRProcess:
    def __init__(self) -> None:
        # self.max_opportunity = 10
        self.enlarge_bbox_ratio = 0.2
        self.max_trigger_slam = 5
        self.data_lists = load_txt("/home/portable-00/VisionCopilot/pharmacy/database/medicine_names.txt")
        self.text_sys = TextSystem(utility.parse_args())
        self.candiancate = []
        
    def _merge_drug_lists(self, list1, list2):
        merged_list = []
        reserve_list = []
        result_list = []
        i,j = 0,0
        if len(list1) > 1 and len(list2) == 1:
            return list1
        while i < len(list1) and j < len(list2):
            if list1[i] == list2[j]:
                merged_list.append(list1[i]) if list1[i] not in merged_list else None
                i += 1
                j += 1
            elif list2[j] in list1 and list1.index(list2[j]) > i:
                if list1.index(list2[j]) == i + 1:
                    merged_list.extend(reserve_list)
                reserve_list = []
                merged_list.append(list1[i])
                i += 1
            else:
                reserve_list.append(list2[j])
                j += 1
        merged_list.extend(list1[i:])
        merged_list.extend(list2[j:])
        merged_list.extend(reserve_list)
        for item in reversed(merged_list):
            result_list.insert(0, item) if item not in result_list else None
        return result_list
    
    def check_boundary(self,bbox_xyxy,image_xyxy,shape):
        bbox_xyxy[0]
    
    def scan_prescription(self):
        end_trigger_times = 0
        max_candicated = 0
        times = 0
        res_counter = []
        result_counter = {}
        frame_collections = {}
        for filename in sorted(os.listdir(r"/home/portable-00/data/images"),key=get_numeric_part):
            times += 1
            frame = cv2.imread(r"/home/portable-00/data/images/"+filename)
            (dt_box_res,prescription,trigger) = procession(frame,self.text_sys,self.data_lists,"prescription")
            end_trigger_times += 1 if trigger else 0
            if max_candicated < prescription.__len__():
                max_candicated = prescription.__len__()
                res_counter = [dt_box_res,prescription]
                res_frame = frame
            # print(prescription)
            # a = time.time()
            for result in prescription:
                result_counter[result] = ClassDict(
                        tickles = times,
                        max_candicated = [dt_box_res,prescription],
                        res_frame = frame,
                        counts = 1
                    ) if result not in result_counter else ClassDict(
                        tickles = times if prescription.__len__() > result_counter[result].max_candicated[1].__len__() else result_counter[result].tickles,
                        max_candicated =  [dt_box_res,prescription] if prescription.__len__() > result_counter[result].max_candicated[1].__len__() else result_counter[result].max_candicated,
                        res_frame = frame if prescription.__len__() > result_counter[result].max_candicated[1].__len__() else result_counter[result].res_frame,
                        counts = result_counter[result].counts + 1
                    )
            # print(time.time() - a)
            self.candiancate = self._merge_drug_lists(self.candiancate,prescription) # Waiting For implemention
        
        frame_collections.update({elements.tickles: ClassDict(frame=elements.res_frame, max_candicated=elements.max_candicated) 
                                  for i, elements in result_counter.items() 
                                  if elements.tickles not in frame_collections})
        
        for tickles,res_pack in frame_collections.items():   
            res_frame , res_counter= res_pack.frame , res_pack.max_candicated
            height,width = getavgSize(res_counter[0])
            # Cover Checked places
            for res in res_counter[0]:
                res_frame = cv2.rectangle(res_frame, tuple(res[0].astype("int")),tuple(res[2].astype("int")),color=(0, 255, 0),thickness=-1)
            conter_len = 0 
            first_set = True
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
                        cv2.imwrite("refe/"+str(tickles)+"_"+str(conter_len)+".jpg",res_frame[selected_height:selected_height + int(height * 1.5),
                                                   selected_width:selected_width + int(width * 1.5)]
                                         )
                        fix_height += height
                        if rec_res == None:
                            fix_height += height
                            continue
                        conter_len += 1
                        res_counter[1].insert(i - 1 + conter_len,rec_res)
            cv2.imwrite(str(tickles)+".jpg",res_frame)
            # print(res_counter[1], tickles)
            # self._merge_drug_lists(self.candiancate,res_counter[1])
        print(res_counter[1])
        print(result_counter)
        print(self.candiancate)
test = OCRProcess()
test.scan_prescription()