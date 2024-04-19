
import time
import numpy as np
from qinglang.utils.utils import Config, load_json
from utils.ocr_infer.load_data_list import load_txt
from utils.ocr_infer.ocr_processor import procession
from qinglang.data_structure.video.video_base import VideoFlow
from utils.yolv_infer.curr_false import curr_false
from utils.yolv_infer.yolov_teller import find_medicine_by_name
from utils.ocr_infer.predict_system import TextSystem
import utils.ocr_infer.pytorchocr_utility as utility
import argparse

class OcrDector:
    def __init__(self) -> None:
        self.source = Config("configs/source.yaml")
        self.configs = Config("configs/ocr/wild_ocr.yaml")
        self.configs.update(Config("configs/source.yaml"))
        self.video_flow = VideoFlow("/home/portable-00/VisionCopilot/test/test_1.mp4")
        self.text_sys = TextSystem(argparse.Namespace(**self.configs.__dict__))
        self.data = load_json(self.source.medicine_database)
        self.data_lists = load_txt(self.source.medicine_names)
        self.reserve_bbox = []
        # self.back_current_shelf = ''

    def ocr_detect(self, frame):
        matching_medicines = []
        ocr_dt_boxes, ocr_rec_res = procession(frame, self.text_sys, self.data_lists, "process")
        for i in range(len(ocr_dt_boxes)):
            matching_medicines.append(find_medicine_by_name(self.data, curr_false(ocr_rec_res[i][0], self.data_lists[:-2])))
        return matching_medicines            
            
        

if __name__ == '__main__':
    ...