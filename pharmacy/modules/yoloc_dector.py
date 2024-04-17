import traceback
import numpy as np
from qinglang.utils.utils import Config, load_json

from dependency.yolo.models.yolo.model import YOLO
from utils.ocr_infer.load_data_list import load_txt
from utils.ocr_infer.ocr_processor import procession
from qinglang.data_structure.video.video_base import VideoFlow
from utils.yolv_infer.curr_false import curr_false
from utils.yolv_infer.yolov_teller import find_medicine_by_name, get_drug_by_index, tensor_converter
from utils.ocr_infer.predict_system import TextSystem
import utils.ocr_infer.pytorchocr_utility as utility


class YolovDector:
    def __init__(self) -> None:
        self.config = Config(
            yolov_path=rf"/home/portable-00/VisionCopilot/pharmacy/checkpoints/yolo/last.pt",
        )
        self.model = YOLO(self.config.yolov_path)
        self.video_flow = VideoFlow("/home/portable-00/VisionCopilot/test/test_1.mp4")
        self.text_sys = TextSystem(utility.parse_args())
        self.data = load_json("/home/portable-00/VisionCopilot/pharmacy/database/medicine_database.json")
        self.data_lists = load_txt("/home/portable-00/VisionCopilot/pharmacy/database/medicine_names.txt")
        self.reserve_bbox = []
        # self.back_current_shelf = ''

    def scan_prescription(self, frame):
        return procession(frame, self.text_sys, self.data_lists, "prescription"), self.data_lists[-2:]

    def yolo_detect(self, frame):
        results = self.model(frame, verbose=False)
        for result in results:
                boxes = result.boxes
                # probs = result.probs
                cls, conf, xywh = boxes.cls, boxes.conf, boxes.xywh
                if cls.__len__() == 0:
                    pass
                else:
                    return [tensor_converter(cls), tensor_converter(xywh)]
         
    def ocr_detect(self, frame):
        matching_medicines = []
        ocr_dt_boxes, ocr_rec_res = procession(frame, self.text_sys, self.data_lists, "process")
        for i in range(len(ocr_dt_boxes)):
            matching_medicines.append(find_medicine_by_name(self.data, curr_false(ocr_rec_res[i][0], self.data_lists[:-2])))
        return matching_medicines            
    
    def detect_medicines(self, frame):
        try:
            # ocr
            matching_medicines = None
            ocr_dt_boxes, ocr_rec_res = procession(frame, self.text_sys, self.data_lists, "process")
            for i in range(len(ocr_dt_boxes)):
                matching_medicines  = find_medicine_by_name(self.data,
                                                           curr_false(ocr_rec_res[i][0], self.data_lists[:-2]))

                if matching_medicines:
                    matching_medicines = matching_medicines
                    pos = i
                    ocr_current_shelf = matching_medicines.get("货架号")  # or other info in the dataset
                    break
                else:
                    ocr_current_shelf = ''
            # yolo
            results = self.model(frame, verbose=True)
            for result in results:
                boxes = result.boxes
                # probs = result.probs
                cls, conf, xywh = boxes.cls, boxes.conf, boxes.xywh  # get info needed
                # print([tensor_converter(cls), tensor_converter(xywh)])
                if cls.__len__() == 0:
                    if matching_medicines is not None:
                        return [[matching_medicines['index'] - 1],self.convert_bbox_to_yolov(ocr_dt_boxes[pos],frame.shape[0],frame.shape[1])]
                    pass
                else:
                    current_drug = get_drug_by_index(int(cls[0]), self.data)
                    if ocr_current_shelf is not None:
                        if current_drug.get("货架号") != ocr_current_shelf:
                            detect_res_cls = "nomatch"
                            return [detect_res_cls, []]
                        else:
                            return [tensor_converter(cls), tensor_converter(xywh)]
                    else:
                        return [tensor_converter(cls), tensor_converter(xywh)]
        except Exception as e:
            traceback.print_exc()

    def drug_match(self, medicine_cls, prescription):
        med_name = get_drug_by_index(medicine_cls, self.data)
        if med_name["药品名称"] in prescription:
            return True
        else:
            return False
    
    def convert_bbox_to_yolov(self,bbox, image_width, image_height):
        yolo_boxes = []
        # 计算边界框的中心点坐标
        center_x = np.mean(bbox[:, 0])
        center_y = np.mean(bbox[:, 1])
    
        # 计算边界框的宽度和高度
        width = np.max(bbox[:, 0]) - np.min(bbox[:, 0])
        height = np.max(bbox[:, 1]) - np.min(bbox[:, 1])
    
        # 将边界框坐标转换为 YOLOv3 格式（绝对像素坐标）
        yolo_x = center_x 
        yolo_y = center_y 
        yolo_width = width * 3
        yolo_height = height * 3
    
            # 将转换后的边界框添加到列表中
        yolo_boxes.append([yolo_x, yolo_y, yolo_width, yolo_height])
    
        return yolo_boxes
    
    def test_yolo(self):
        for frame_id, frame in enumerate(self.video_flow):
            result = self.yolo_detect(frame)
            print(result)     
            
if __name__ == '__main__':
    test1 = YolovDector()
    test1.test_yolo()