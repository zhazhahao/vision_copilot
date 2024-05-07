from collections import Counter
import re
import Levenshtein
import cv2
from qinglang.utils.utils import Config, load_json
from utils.ocr_infer.load_data_list import load_txt
import numpy as np
from utils.ocr_infer.prescription_utils import FrameMaxMatchingCollections, PreScriptionRecursiveObject
from utils.yolv_infer.index_transfer import IndexTransfer
from utils.yolv_infer.curr_false import curr_false
from utils.ocr_infer.predict_system import TextSystem
from utils.utils import MedicineDatabase
import argparse


class OcrDector:
    def __init__(self) -> None:
        self.database = MedicineDatabase()
        self.source = Config("configs/source.yaml")
        self.configs = Config("configs/ocr/wild_ocr.yaml")
        self.configs.update(Config("configs/source.yaml"))
        self.indexTransfer = IndexTransfer()
        self.max_opportunity = 10
        self.enlarge_bbox_ratio = 0.2
        self.loss_track_threshold = 60
        self.frame_collections = FrameMaxMatchingCollections()
        self.candiancate = PreScriptionRecursiveObject()
        self.text_sys = TextSystem(argparse.Namespace(**self.configs.__dict__))
        self.data = load_json(self.source.medicine_database)
        self.data_lists = load_txt(self.source.medicine_names)
        self.end_trigger_times = 0
        self.loss_track_times = 0
        self.times = 0
        self.reserve_bbox = []
        self.status_dict = {}
            
    def scan_prescription(self,frame):
        res_counter = []
        counter = Counter()
        for frame in stream:
            self.times += 1
            if not self.finish_candidate:
                (dt_box_res,prescription,trigger) = self.ocr_detector.procession(frame,"prescription")
                if "领退药药单汇总" in prescription or "统领单(针剂)汇总" in prescription:
                    self.candiancate.update(prescription,self.times)
                    self.finish_candidate = True
                continue
            (dt_box_res,prescription,trigger) = self.ocr_detector.procession(frame,"prescription")
            self.end_trigger_times += 1 if trigger else 0
            counter.update(prescription)
            res_frame , res_counter= frame , [dt_box_res,prescription]
            height,width = self.ocr_detector.getavgSize(res_counter[0])
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
                            selected_height = res_frame.shape[0] - int(height * 1.5)
                        rec_res = self.ocr_detector.procession(res_frame[selected_height + int(height * 0.5):selected_height + int(height * 1.5),
                                                   selected_width:selected_width + int(width * 1.5)],options="Single")
                        # print(prescription)
                        self.status_dict[rec_res] = "Update"
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
                if selected_height - int(height * 1) < 0:
                    selected_height = int(height)
                if selected_width - int(width) < 0:
                    selected_width = int(width)
                if selected_width + int(width) > 1920:
                    selected_width = 1920 - int(width)

                rec_res = self.ocr_detector.procession(res_frame[selected_height - int(height * 1):selected_height ,
                                           selected_width - int(width):selected_width + int(width)]
                                 ,options="Single")
                if rec_res != None:
                    res_counter[1].insert(0,rec_res)
            self.frame_collections.update(frame,max_candicated=[dt_box_res,prescription],times=self.times)
            # print(self.end_trigger_times)  
            # print(loss_track_threshold)
            if res_counter[1].__len__() > 0:
                self.candiancate.update(res_counter[1],self.times)
                loss_track_times = 0
            else:
                loss_track_times += 1
            if loss_track_times > self.loss_track_threshold:
                break
            if self.end_trigger_times == self.max_opportunity:
                # print(self.times)
                break
        # print(prescription)
        tools = self.frame_collections.values()
        for key,values in tools.items():
            self.candiancate._check_merge_drugs([key,values["max_candicated"][1]])
        try:
            max_count = np.median([value for value in counter.values() if value != "领退药药单汇总"])
            max_counts = np.median([answer.counts for answer in self.frame_collections.result_counter.values()])
        except:
            return None
        recursive_con = [answer 
            for answer in self.candiancate.recursive_obj
           if counter[answer] >= max_count / 4 or (answer in self.status_dict and self.frame_collections.result_counter[answer].counts >= max_counts / 4)]
     
        final_con = [answer 
                    for obj in self.candiancate.static_obj  
                    for answer in obj[1]
                   if counter[answer] >= max_count / 4 or (answer in self.status_dict and self.frame_collections.result_counter[answer].counts >= max_counts / 4)]
        
        recursive_con.extend(final_con)
        answer_dict = self.ocr_detector.group_similar_strings(list(set(recursive_con)),self.frame_collections.result_counter)
        return [ans for answer_res in answer_dict
                for ans in answer_res if ans != "统领单(针剂)汇总" or ans != "领退药药单汇总"]
    
    def ocr_detect(self, frame):
        ocr_dt_boxes, ocr_rec_res = self.text_sys(frame)
        # ocr_dt_boxes, ocr_rec_res = self.procession(frame, "process")
        matching_medicines = [{"category_id":answer["Category ID"], "bbox":self.bbox_to_xywh(ocr_dt_boxes[i])}
            for i in range(len(ocr_dt_boxes)) if curr_false(ocr_rec_res[i][0], self.data_lists[:-2]) is not None
            for answer in self.database.find(curr_false(ocr_rec_res[i][0], self.data_lists[:-2]))]
        # print(matching_medicines)
        return matching_medicines 

    def bbox_to_xywh(self,bbox):
        return np.array([np.mean(bbox[:, 0]), np.mean(bbox[:, 1]), np.max(bbox[:, 0]) - np.min(bbox[:, 0]),np.max(bbox[:, 1]) - np.min(bbox[:, 1])])


    def group_similar_strings(self,strings, counter):
        groups = []
        for string in strings:
            # Check if the string can be added to any existing group
            added = False
            for group in groups:
                for s in group:
                    common_chars = set(string) & set(s)
                    if len(common_chars) > 6:
                        group.append(string)
                        added = True
                        break
                if added:
                    break
            if not added:
                groups.append([string])

        # Filter out elements with smaller counts based on the given counter
        filtered_groups = []
        for group in groups:
            filtered_group = []
            max_num = max([counter[element]["counts"] for idx,element in enumerate(group)])
            for idx, element in enumerate(group):
                if counter[element]["counts"]/max_num >= 0.2:  # Adjust the threshold as needed
                    filtered_group.append(element)
            if filtered_group:
                filtered_groups.append(filtered_group)
        return filtered_groups

    def curr_false(self,text,max_threshold=0.4):
        search_ch_text = re.compile(r'[\u4e00-\u9fff]')
        search_brackets_text = re.compile(r'[()\[\]\u3000\uFF08\uFF09\u3010\u3011]')

        if search_ch_text.search(text):
            if text in self.data_lists:
                return text
            
            # while searching number，you should do some improvements.
            text = re.sub(r'\d+[^0-9]*$', '', text)
            filtered_data_lists = self.data_lists.copy()
            
            if search_brackets_text.search(text):
                filtered_data_lists = [item for item in filtered_data_lists if search_brackets_text.search(item[0])]
                
            text = text.replace("（", "(").replace("）", ")").replace("【", "[").replace("】", "]")
            
            similarities = [Levenshtein.ratio(text, str2[0]) for str2 in filtered_data_lists]
            max_similarity = max(similarities)
            max_index = similarities.index(max_similarity)
            most_similar_drug = filtered_data_lists[max_index][0]

            return most_similar_drug if max_similarity > max_threshold else None

    
if __name__ == '__main__':
    ...