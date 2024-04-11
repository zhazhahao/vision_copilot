import json
import traceback
import cv2
from modules.camera_processor import CameraProcessor
from modules.ocr_processor import procession
import utils.ocr_infer.pytorchocr_utility as utility
from utils.yolv_infer.yolov_teller import tensor_converter ,get_drug_by_index ,find_medicine_by_name
from utils.yolv_infer.curr_false import curr_false
from dependency.yolo import YOLO
import threading

video_stream_path = 'rtsp://192.168.3.100/live'  # 在这里更接受推流的地址
cv2.CAP_PROP_READ_TIMEOUT_MSEC = 1e3
# tryToConnect = [1, 2, 3]  #
none_detection = None
model = YOLO("pharmacy/models/last.pt")
ocr_current_shelf = ''

# 打开药品数据库
with open("pharmacy/data/medicine_database.json", 'r', encoding='utf-8') as file:
    try:
        data = json.load(file)
    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)
        exit()


def data_list():
    data_list=[]
    with open("pharmacy/data/medicine_names.txt", 'r', encoding='utf-8') as file:
        lines = file.readlines()
    for line in lines:
        # 去除行尾的换行符并按空格切分
        items = line.strip().split()
        data_list.append(items)
        # 使用第一列元素作为键，整行作为对应的值
    return data_list
data_lists = data_list()

def yolo_and_ocr_0(frame, model,text_sys):
    global ocr_current_shelf
    try:
        # ocr
        ocr_dt_boxes, ocr_rec_res = procession(frame, text_sys,data_lists,"process")
        for i in range(len(ocr_dt_boxes)):
            matching_medicines = find_medicine_by_name(data, curr_false(ocr_rec_res[i][0],data_lists))
            if matching_medicines:
                ocr_current_shelf = matching_medicines.get("货架号")  # or other info in the dataset
        # yolo
        results = model(frame)
        for result in results:
            boxes = result.boxes
            # probs = result.probs
            cls, conf, xywh = boxes.cls, boxes.conf, boxes.xywh  # get info needed
            print([tensor_converter(cls), tensor_converter(xywh)])
            if cls.__len__()==0:
                pass
            else:
                current_drug = get_drug_by_index(int(cls[0]),data)

                if current_drug.get("货架号") != ocr_current_shelf:
                    detect_res_cls = "nomatch"
                    return [detect_res_cls, []]

                else:
                    detect_res_cls = cls
                    return [tensor_converter(detect_res_cls), tensor_converter(xywh)]
    except Exception as e:
        traceback.print_exc()