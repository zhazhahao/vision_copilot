from qinglang.utils.utils import Config, load_json
from utils.ocr_infer.load_data_list import load_txt
from utils.ocr_infer.ocr_processor import procession
import numpy as np
from qinglang.data_structure.video.video_base import VideoFlow
from utils.yolv_infer.index_transfer import IndexTransfer
from utils.yolv_infer.curr_false import curr_false
from utils.yolv_infer.yolov_teller import find_medicine_by_name
from utils.ocr_infer.predict_system import TextSystem
import argparse


class OcrDector:
    def __init__(self) -> None:
        self.source = Config("configs/source.yaml")
        self.configs = Config("configs/ocr/wild_ocr.yaml")
        self.configs.update(Config("configs/source.yaml"))
        self.indexTransfer = IndexTransfer()
        self.text_sys = TextSystem(argparse.Namespace(**self.configs.__dict__))
        self.data = load_json(self.source.medicine_database)
        self.data_lists = load_txt(self.source.medicine_names)
        self.reserve_bbox = []

    def ocr_detect(self, frame):
        ocr_dt_boxes, ocr_rec_res = procession(frame, self.text_sys, self.data_lists, "process")
        matching_medicines = [{"category_id":answer,"bbox":self.bbox_to_xywh(ocr_dt_boxes[i])}
            for i in range(len(ocr_dt_boxes)) if curr_false(ocr_rec_res[i][0], self.data_lists[:-2]) is not None
            for answer in self.indexTransfer.name2cls(self.data_lists,curr_false(ocr_rec_res[i][0], self.data_lists[:-2]))]
        return matching_medicines 

    def bbox_to_xywh(self,bbox):
        return np.array([np.mean(bbox[:, 0]), np.mean(bbox[:, 1]), np.max(bbox[:, 0]) - np.min(bbox[:, 0]),np.max(bbox[:, 1]) - np.min(bbox[:, 1])])

if __name__ == '__main__':
    ...