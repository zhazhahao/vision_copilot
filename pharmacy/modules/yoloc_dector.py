import cv2
import numpy as np
from qinglang.utils.utils import Config, load_json

from utils.ocr_infer.load_data_list import load_txt
from dependency.yolo.models.yolo.model import YOLO
from qinglang.data_structure.video.video_base import VideoFlow
from utils.yolv_infer.yolov_teller import get_drug_by_index, tensor_converter


print('YolovDector initialized')

class YolovDector:
    def __init__(self) -> None:
        self.source = Config("configs/source.yaml")
        self.model = YOLO(self.source.yolov_path)

    def yolo_detect(self, frame):
        results = self.model(frame, verbose=False)
        for result in results:
                boxes = result.boxes
                cls, conf, xywh = boxes.cls, boxes.conf, boxes.xywh
                if cls.__len__() == 0:
                    pass
                else:
                    return [tensor_converter(cls), tensor_converter(xywh)]       


if __name__ == '__main__':
    ...