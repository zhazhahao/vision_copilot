import re
import Levenshtein
from qinglang.utils.utils import Config, load_json
import torch
from utils.ocr_infer.load_data_list import load_txt
from utils.ocr_infer.ocr_processor import procession
import numpy as np
from qinglang.data_structure.video.video_base import VideoFlow
from utils.yolv_infer.index_transfer import IndexTransfer
from utils.yolv_infer.curr_false import curr_false
from utils.utils import MedicineDatabase
from utils.yolv_infer.yolov_teller import find_medicine_by_name
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
        self.text_sys = TextSystem(argparse.Namespace(**self.configs.__dict__))
        self.data = load_json(self.source.medicine_database)
        self.data_lists = load_txt(self.source.medicine_names)
        self.reserve_bbox = []

    def ocr_detect(self, frame):
        ocr_dt_boxes, ocr_rec_res = self.procession(frame, "process")
        matching_medicines = [{"category_id":answer["Category ID"], "bbox":self.bbox_to_xywh(ocr_dt_boxes[i])}
            for i in range(len(ocr_dt_boxes)) if curr_false(ocr_rec_res[i][0], self.data_lists[:-2]) is not None
            for answer in self.database.find(curr_false(ocr_rec_res[i][0], self.data_lists[:-2]))]
        return matching_medicines 

    def bbox_to_xywh(self,bbox):
        return np.array([np.mean(bbox[:, 0]), np.mean(bbox[:, 1]), np.max(bbox[:, 0]) - np.min(bbox[:, 0]),np.max(bbox[:, 1]) - np.min(bbox[:, 1])])

    def procession(self,img, options="process"):
        prescription_res = []
        dt_boxes_res = []
        keywords = []
        
        with torch.no_grad():
            if options != "Single":
                dt_boxes, rec_res = self.text_sys(img)
                # print(rec_res)
                if options == "prescription":
                    trigger = False
                    for i, (text, score) in enumerate(rec_res):
                        trigger = True if "合计" in text or trigger == True else False

                        text = curr_false(text,  self.data_lists,0.6)
                        rec_res[i] = (text, score)
                        if text is not None:
                            dt_boxes_res.append(dt_boxes[i])
                            prescription_res.append(text)
                    return dt_boxes_res,prescription_res,trigger
                else:
                    return dt_boxes, rec_res
            else:
                rec_res, predict_time =  self.text_sys.text_recognizer([img])
                rec_res=self.curr_false(rec_res[0][0], 0.6)
                return rec_res
            
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