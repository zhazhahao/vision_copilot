
import traceback
from qinglang.utils.utils import Config,load_json,load_txt


from dependency.yolo.models.yolo.model import YOLO
from modules.ocr_processor import procession

from utils.yolv_infer.curr_false import curr_false
from utils.yolv_infer.yolov_teller import find_medicine_by_name, get_drug_by_index, tensor_converter
from utils.ocr_infer.predict_system import TextSystem
import utils.ocr_infer.pytorchocr_utility as utility

class YolovDector:
    def __init__(self) -> None:
        self.config = Config(
            yolov_path = rf"/home/portable-00/VisionCopilot/pharmacy/checkpoints/yolo/last.pt",
        )
        self.model = YOLO("pharmacy/models/last.pt")
        self.text_sys = TextSystem(utility.parse_args())
        self.data = load_json("/home/portable-00/VisionCopilot/pharmacy/database/medicine_database.json")
        self.data_lists = load_txt("/home/portable-00/VisionCopilot/pharmacy/database/medicine_names.txt")
        self.back_current_shelf = ''
        
    def scan_prescription(self,frame):
        return procession(frame,self.text_sys,self.data_lists,"prescription")
        
    def detect_medicines(self,frame):
        try:
            # ocr
            ocr_dt_boxes, ocr_rec_res = procession(frame, self.text_sys,self.data_lists,"process")
            for i in range(len(ocr_dt_boxes)):
                matching_medicines = find_medicine_by_name(self.data, curr_false(ocr_rec_res[i][0],self.data_lists[:-2]))
                if matching_medicines:
                    ocr_current_shelf = matching_medicines.get("货架号")  # or other info in the dataset
                    self.back_current_shelf = ocr_current_shelf
                else:
                    ocr_current_shelf = ''
            # yolo
            results = self.model(frame)
            for result in results:
                boxes = result.boxes
                # probs = result.probs
                cls, conf, xywh = boxes.cls, boxes.conf, boxes.xywh  # get info needed
                print([tensor_converter(cls), tensor_converter(xywh)])
                if cls.__len__()==0:
                    pass
                else:
                    current_drug = get_drug_by_index(int(cls[0]),self.data)
                    if ocr_current_shelf is not None:
                        if current_drug.get("货架号") != ocr_current_shelf:
                            detect_res_cls = "nomatch"
                            return [detect_res_cls, []]
                        else:
                            return [tensor_converter(detect_res_cls), tensor_converter(xywh)]
                    elif current_drug.get("货架号") != self.back_current_shelf:
                        detect_res_cls = "nomatch"
                        return [detect_res_cls, []]
                    else:
                        detect_res_cls = cls
                        return [tensor_converter(detect_res_cls), tensor_converter(xywh)]
        except Exception as e:
            traceback.print_exc()
    def drug_match(self, medicine_cls, prescription):
        med_name = get_drug_by_index(medicine_cls,self.data)
        if med_name["ҩƷ��"] in prescription:
            return True
        else:
            return False