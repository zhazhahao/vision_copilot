import re
import Levenshtein
import cv2
import numpy as np
from utils.ocr_infer.load_data_list import load_txt
from utils.yolv_infer.index_transfer import IndexTransfer
from qinglang.utils.utils import ClassDict, load_json
from PIL import Image, ImageDraw, ImageFont

class MedicineDatabase:
    def __init__(self) -> None:
        self.data = load_json('database/medicine_database_en.json')

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, key: int) -> ClassDict:
        return self.data[key - 1]

    def find(self, name: str, **kwargs) -> int:
        candidates = [medicine for medicine in self.data if medicine['Name'] == name]
        
        for key, value in kwargs.items():
            candidates = [medicine for medicine in candidates if medicine.get(key, None) == value]
            
        return candidates

def curr_false(self, text, max_threshold=0.4):
    search_ch_text = re.compile(r'[\u4e00-\u9fff]')
    search_brackets_text = re.compile(r'[()\u3000\uFF08\uFF09\u3010\u3011]')
    # search_num_text = re.compile(r'\d')
    if search_ch_text.search(text):
        if text in self.data_lists:
            return text
        # while searching number，you should do some improvements.
        text = re.sub(r'\d+[^0-9]*$', '', text)
        filtered_data_lists = self.data_lists.copy()
        if search_brackets_text.search(text):
            filtered_data_lists = [item for item in filtered_data_lists if search_brackets_text.search(item[0])]
        similarities = [Levenshtein.ratio(text, str2[0]) for str2 in filtered_data_lists]
        max_similarity = max(similarities)
        max_index = similarities.index(max_similarity)
        most_similar_drug = filtered_data_lists[max_index][0]
        if max_similarity > max_threshold:
            return most_similar_drug

def tensor_converter(tensor):
    tensor = tensor.cpu()
    numpy_array = tensor.numpy()
    values_list = numpy_array.tolist()
    return values_list

def drug_match(self, medicine_cls, prescription):
    med_name = self.index_transfer.get_drug_by_index(medicine_cls)
    if med_name["药品名称"] in prescription:
        return True
    else:
        return False

def convert_bbox_to_yolov(bbox, image_width, image_height):
    yolo_boxes = []
    # 计算边界框的中心点坐标
    center_x = np.mean(bbox[:, 0])
    center_y = np.mean(bbox[:, 1])

    # 计算边界框的宽度和高度
    width = np.max(bbox[:, 0]) - np.min(bbox[:, 0])
    height = np.max(bbox[:, 1]) - np.min(bbox[:, 1])

    # 将边界框坐标转换为 YOLO 格式（绝对像素坐标）
    yolo_x = center_x 
    yolo_y = center_y 
    yolo_width = width * 3
    yolo_height = height * 3

    # 将转换后的边界框添加到列表中
    yolo_boxes.append([yolo_x, yolo_y, yolo_width, yolo_height])
    return yolo_boxes    
      
def plot_save(image, xywh, color, thickness: int = 2, save_path: str = None, text: str = None, font_scale: float = 1, font_face: int = cv2.FONT_HERSHEY_SIMPLEX, thickness_text: int = 1, font_path: str = '/home/portable-00/VisionCopilot/pharmacy/dependency/yolo/SimHei.ttf'):  
    
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