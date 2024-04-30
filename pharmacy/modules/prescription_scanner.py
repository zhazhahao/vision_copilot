
from collections import Counter
import multiprocessing
import re

import cv2
import numpy as np
import torch

from utils.ocr_infer.prescription_utils import FrameMaxMatchingCollections, PreScriptionRecursiveObject
from utils.utils import curr_false


class PrescriptionScanner(multiprocessing.Process):
    def __init__(self) -> None:   
        self.max_opportunity = 10
        self.enlarge_bbox_ratio = 0.2
        self.loss_track_threshold = 60
        self.frame_collections = FrameMaxMatchingCollections()
        self.candiancate = PreScriptionRecursiveObject()
        self.finish_candidate = False
        self.end_trigger_times = 0
        self.times = 0
        self.loss_track_threshold = 0
        self.status_dict = {}
        self.res_counter = []
        self.counter = Counter()
        super().__init__()

    def scan_prescription(self,stream):
        for frame in stream:
            self.times += 1
            if not self.finish_candidate:
                (dt_box_res,prescription,trigger) = self.procession(frame,"prescription")
                # print(prescription)
                if "领退药药单汇总" in prescription or "统领单(针剂)汇总" in prescription:
                    self.candiancate.update(prescription,self.times)
                    self.finish_candidate = True
                continue
            (dt_box_res,prescription,trigger) = self.procession(frame,"prescription")
            self.end_trigger_times += 1 if trigger else 0
            self.counter.update(prescription)
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
                            selected_height = res_frame.shape[0] - int(height * 1.5)
                        rec_res = self.procession(res_frame[selected_height + int(height * 0.5):selected_height + int(height * 1.5),
                                                   selected_width:selected_width + int(width * 1.5)],options="Single")
                        # print(prescription)
                        self.status_dict[rec_res] = "Update"
                        fix_height += height
                        if rec_res == None:
                            fix_height += height
                            continue
                        conter_len += 1
                        self.counter.update([rec_res])
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

                rec_res = self.procession(res_frame[selected_height - int(height * 1):selected_height ,
                                           selected_width - int(width):selected_width + int(width)]
                                 ,options="Single")
                if rec_res != None:
                    res_counter[1].insert(0,rec_res)
            self.frame_collections.update(frame,max_candicated=[dt_box_res,prescription],times= self.times)
            # print(end_trigger_times)  
            # print(loss_track_threshold)
            if res_counter[1].__len__() > 0:
                self.candiancate.update(res_counter[1],self.times)
                loss_track_threshold = 0
            else:
                loss_track_threshold += 1
            if loss_track_threshold > self.loss_track_threshold:
                break
            if self.end_trigger_times == self.max_opportunity:
                # print(times)
                break
        # print(prescription)
        tools = self.frame_collections.values()
        for key,values in tools.items():
            self.candiancate._check_merge_drugs([key,values["max_candicated"][1]])
        try:
            max_count = np.median([value for value in self.counter.values() if value != "领退药药单汇总"])
            max_counts = np.median([answer.counts for answer in self.frame_collections.result_counter.values()])
        except:
            return None
        recursive_con = [answer 
            for answer in self.candiancate.recursive_obj
           if self.counter[answer] >= max_count / 4 or (answer in self.status_dict and self.frame_collections.result_counter[answer].counts >= max_counts / 4)]
     
        final_con = [answer 
                    for obj in self.candiancate.static_obj  
                    for answer in obj[1]
                   if self.counter[answer] >= max_count / 4 or (answer in self.status_dict and self.frame_collections.result_counter[answer].counts >= max_counts / 4)]
        
        recursive_con.extend(final_con)
        answer_dict = self.ocr_detector.group_similar_strings(list(set(recursive_con)),self.frame_collections.result_counter)
        # print(answer_dict)
        # print(ans for answer_res in answer_dict
        #         for ans in answer_res)
        return [ans for answer_res in answer_dict
                for ans in answer_res if ans != "统领单(针剂)汇总" or ans != "领退药药单汇总"]

    def getavgSize(self,dt_boxes):
        if dt_boxes is not None and len(dt_boxes) != 0:
            rects = np.array(dt_boxes)
            heights = rects[:, 3, 1] - rects[:, 0, 1] + rects[:, 2, 1] - rects[:, 1, 1]
            widths = rects[:, 2, 0] - rects[:, 0, 0] - rects[:, 3, 0] + rects[:, 1, 0]
            return heights.max()/2 , widths.mean()/2
        else:
            return 0, 0  
    
    def procession(self,rec_ress, options="process"):
        prescription_res = []
        dt_boxes_res = []
        call_box = []
        with torch.no_grad():
            if options != "Single":
                dt_boxes, rec_res = rec_ress
                # print(rec_res)
                if options == "prescription":
                    trigger = False
                    for i, (text, score) in enumerate(rec_res):
                        trigger = True if "合计" in text or trigger == True else False
                        
                        regex = re.compile("集采|/")
                        # 在文本中查找匹配的关键字
                        
                        print(text)
                        matches = regex.search(text)
                        if matches and score >= 0.8:
                            text.replace(" ", "")
                            # print(text)
                            call_box.append(text)  
                        pre_text = text      
                        try:
                            data_lis = [data for data in self.data_lists if "氨" in data[0] and data[0].index("氨") == text.index("氨")]
                            text = curr_false(text,data_lis,0.9)
                        except:
                            text = curr_false(text, self.data_lists,0.8)
                        rec_res[i] = (text, score)
                        if text is not None:
                            dt_boxes_res.append(dt_boxes[i])
                            prescription_res.append(text)
                    print(call_box)
                    return dt_boxes_res,prescription_res,trigger
                else:
                    return dt_boxes, rec_res
            else:
                rec_res, predict_time =  self.text_sys.text_recognizer([img])
                rec_res=self.curr_false(rec_res[0][0], 0.6)
                return rec_res