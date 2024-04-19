import traceback
import cv2
import numpy as np
from qinglang.utils.utils import Config, load_json
from PIL import Image, ImageDraw, ImageFont
from utils.ocr_infer.load_data_list import load_txt
from utils.ocr_infer.ocr_processor import procession
from dependency.yolo.models.yolo.model import YOLO
from qinglang.data_structure.video.video_base import VideoFlow
from utils.yolv_infer.curr_false import curr_false
from utils.yolv_infer.yolov_teller import find_medicine_by_name, get_drug_by_index, tensor_converter
from utils.ocr_infer.predict_system import TextSystem
import utils.ocr_infer.pytorchocr_utility as utility


class YolovDector:
    def __init__(self) -> None:
        self.source = Config("configs/source.yaml")

        self.model = YOLO(self.source.yolov_path)
        self.video_flow = VideoFlow(self.source.virtual_camera_source)
        self.text_sys = TextSystem(utility.parse_args())
        self.data = load_json(self.source.medicine_database)
        self.data_lists = load_txt(self.source.medicine_names)
        self.reserve_bbox = []
        # self.back_current_shelf = ''

    # def scan_prescription(self, frame):
    #     return procession(frame, self.text_sys, self.data_lists, "prescription"), self.data_lists[-2:]

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
    
    # def test_yolo(self):
    #     for frame_id, frame in enumerate(self.video_flow):
    #         result = self.yolo_detect(frame)
    #         print(result)     
      
    def plot_save(self, image, xywh, color, thickness: int = 2, save_path: str = None, text: str = None, font_scale: float = 1, font_face: int = cv2.FONT_HERSHEY_SIMPLEX, thickness_text: int = 1, font_path: str = '/home/portable-00/VisionCopilot/pharmacy/dependency/yolo/SimHei.ttf'):  
        
        pilimg = Image.fromarray(image)    
        draw = ImageDraw.Draw(pilimg)  # 图片上打印
        font = ImageFont.truetype(font_path, int(font_scale * 20), encoding='utf-8')  # PIL requires the size to be an int
        if text:
            x0, y0, x1, y1  = font.getbbox(text)
            text_height = y1 - y0 
            # 设置文本的左上角位置，边界框的左上角偏移文本高度
            text_pt = (xywh[0], xywh[1] - text_height - thickness_text)
            draw.text(text_pt, text, color, font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
        
        cv2charimg = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
        cv2.rectangle(cv2charimg, pt1=xywh[:2], pt2=[xywh[0]+xywh[2], xywh[1]+xywh[3]], color=color, thickness=thickness)
        if save_path:
            cv2.imwrite(save_path, cv2charimg)
            
        

if __name__ == '__main__':
    ...