from collections import Counter
import os
import time
import cv2
import numpy as np
import utils.ocr_infer.pytorchocr_utility as utility
from utils.ocr_infer.predict_system import TextSystem
from utils.ocr_infer.ocr_processor_now import procession
from utils.ocr_infer.load_data_list import load_txt
from qinglang.utils.utils import ClassDict

from utils.utils import MedicineDatabase
from utils.yolv_infer.curr_false import group_similar_strings



class PreScriptionRecursiveObject:
    def __init__(self) -> None:
        self.from_index = 0
        self.to_index = 0 
        self.times = 0
        self.recursive_epoch = 3
        self.recursive_obj = []
        self.static_obj = []
        self.quest_refer = {}
        self.quest_loss = []
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
        if len(reserve_list) > len(list2) - len(reserve_list) and (self.static_obj.__len__() == 0 or times - self.static_obj[self.static_obj.__len__() - 1][0] > 10):
            print(times)
            self.static_obj.append([times,merged_list])
            merged_list = []
        merged_list.extend(reserve_list)
        # for item in reversed(merged_list):
        #     result_list.insert(0, item) if item not in result_list else None
        return merged_list
    
    def _check_merge_drugs(self,image):
        if self.static_obj.__len__():
            for i in range(len(self.static_obj)):
                if self.static_obj[i][0] >= image[0]:
                    break
            j = 0
            print(self.static_obj[i][1],image[1],self.find_positions(self.static_obj[i][1],
                                      image[1]))
            
    def find_positions(self,arr1, arr2):
        positions = []
        used_indexes = set()  # 记录已经使用过的索引
        for item in arr1:
            found = False
            for i, x in enumerate(arr2):
                if x == item and i not in used_indexes:
                    positions.append(i)
                    used_indexes.add(i)
                    found = True
                    break
            if not found:
                positions.append(None)
        return positions                                       

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
        self.test = False
        self.max_opportunity = 10
        self.loss_track_threshold = 30
        self.enlarge_bbox_ratio = 0.2
        self.data_lists = load_txt("/home/portable-00/VisionCopilot/pharmacy/database/medicine_names.txt")
        self.text_sys = TextSystem(utility.parse_args())
        self.candiancate = PreScriptionRecursiveObject()
        self.medicine_database = MedicineDatabase()
        self.frame_collections = FrameMaxMatchingCollections()
        self.finish_candidate = False
        self.volumns = [i["Specification"].replace(" ", "") for i in self.medicine_database]
    
    def scan_prescription(self):
        end_trigger_times = 0
        times = 0
        data_base = {}
        lose_track = 0
        status_dict = {}
        res_counter = []
        counter = Counter()
        for filename in sorted(os.listdir(r"/home/portable-00/data/images"),key=self.get_numeric_part):
            times += 1
            frame = cv2.imread(r"/home/portable-00/data/images/"+filename)
            if self.finish_candidate:
                (dt_box_res,prescription,trigger,call_box) = procession(frame,self.text_sys,self.data_lists,"prescription")
                if "领退药药单汇总" in prescription or "统领单(针剂)汇总" in prescription:
                    self.candiancate.update(prescription,times)
                    self.finish_candidate = True
                
            (dt_box_res,prescription,trigger,volumn_counter) = procession(frame,self.text_sys,self.data_lists[:-2],"prescription")
            for volumn in volumn_counter:
                if volumn in self.volumns:
                    volumn = volumn.replace("m1","ml").replace("Ml","ml")
                    data_base.update({x:self.medicine_database[x] for x in range(self.volumns.__len__()) if self.volumns[x] == volumn or volumn.__len__() > 4 and(volumn in self.volumns[x] )})
                    # print(self.medicine_database[self.volumns.index(volumn)])
            end_trigger_times += 1 if trigger else 0
            counter.update(prescription)
            res_frame , res_counter= frame , [dt_box_res,prescription]
            height,width = self.getavgSize(res_counter[0])
            for res in res_counter[0]:
                res_frame = cv2.rectangle(res_frame, tuple(res[0].astype("int")),tuple(res[2].astype("int")),color=(0, 255, 0),thickness=-1)
            conter_len = 0 
            min_width = 1920
            for i in range(1, len(res_counter[0])):
                if (res_counter[0][i][3][1] - res_counter[0][i-1][0][1]) >= height * 1.5:
                    fix_height = res_counter[0][i-1][0][1]
                    first_set = True
                    while fix_height < res_counter[0][i][0][1]:
                        selected_height = min(int(res_counter[0][i - 1][2][1]),int(res_counter[0][i - 1][3][1])) if first_set else int(selected_height + height)
                        selected_width  = min(int(res_counter[0][i - 1][3][0]),int(res_counter[0][i][3][0]))
                        min_width = min(min_width,selected_width)
                        first_set = False
                        if selected_height + int(height * 1.5) > res_frame.shape[0]:
                            selected_height = res_frame.shape[0]
                        rec_res = procession(res_frame[selected_height + int(height * 0.5):selected_height + int(height * 1.5),
                                                   selected_width:selected_width + int(width * 1.5)]
                                         ,self.text_sys,data_lists=self.data_lists[:-2],options="Single")
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
            if res_counter[0].__len__() != 0:
                selected_height = min(int(res_counter[0][0][0][1]),int(res_counter[0][0][1][1])) 
                selected_width  = int(res_counter[0][0][2][0])                         
                if selected_height - int(height) * 1.5 < 0:
                    continue
                rec_res = procession(res_frame[selected_height - int(height * 1):selected_height ,
                                           selected_width - int(width):selected_width + int(width)]
                                 ,self.text_sys,data_lists=self.data_lists[:-2],options="Single")
                
                if self.test:
                    cv2.imwrite("refe/"+str(filename)+"_"+str(conter_len)+".jpg",res_frame[selected_height - int(height * 1.5):selected_height ,
                                               selected_width - int(width):selected_width + int(width)]
                                         )
                if rec_res != None:
                    # print(rec_res,filename)
                    res_counter[1].insert(0,rec_res)
            # print(res_counter[1])
            self.frame_collections.update(frame,max_candicated=[dt_box_res,prescription],times=times)

            cv2.imwrite(str(times)+".jpg",res_frame)
            if res_counter[1].__len__() > 0:
                self.candiancate.update(res_counter[1],times)
            if end_trigger_times == self.max_opportunity:
                # print(times)
                break
            if res_counter[1].__len__() == 0:
                lose_track += 1
                if self.loss_track_threshold < lose_track:
                    print("No tracking , pass down the precess")
            else:
                lose_track = 0
                
        tools = self.frame_collections.values()
        for key,values in tools.items():
            self.candiancate._check_merge_drugs([key,values["max_candicated"][1]])
        # res_con = [answer for answer in self.candiancate.recursive_obj  
        #            if not(answer in counter.keys() and counter[answer] == 1 ) or answer 
        #            in status_dict.keys()]
        final_con = [answer 
                    for obj in self.candiancate.static_obj  
                    for answer in obj[1]
                   if counter[answer] >= 5 or (answer in status_dict and self.frame_collections.result_counter[answer].counts >= 5)]
        # dict_all = set(final_con)
        answer_dict = group_similar_strings(list(set(final_con)),self.frame_collections.result_counter)
        print(self.frame_collections.result_counter)
        print(status_dict)
        print(self.candiancate.recursive_obj)
        res_dict =[ans for answer_res in answer_dict
                for ans in answer_res]
        print([data["Name"] if data["Name"] in res_dict else None for data in data_base.values()])
        return [ans for answer_res in answer_dict
                for ans in answer_res]
        
    def getavgSize(self,dt_boxes):
        if dt_boxes is not None and len(dt_boxes) != 0:
            rects = np.array(dt_boxes)
            heights = rects[:, 3, 1] - rects[:, 0, 1] + rects[:, 2, 1] - rects[:, 1, 1]
            widths = rects[:, 2, 0] - rects[:, 0, 0] - rects[:, 3, 0] + rects[:, 1, 0]
            return heights.max()/2 , widths.mean()/2
        else:
            return 0, 0
        
    def get_numeric_part(self,filename):
        numeric_part = ''.join(filter(str.isdigit, filename))
        return int(numeric_part) if numeric_part else 0
        
test = OCRProcess()
test.scan_prescription()