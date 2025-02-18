import time
import torch
from utils.utils import MedicineDatabase

def procession(img, text_sys, data_lists, options="process"):
    prescription_res = []
    dt_boxes_res = []
    keywords = []
    with torch.no_grad():
        if options != "Single":
            dt_boxes, rec_res = text_sys(img)
            # print(rec_res)
            if options == "prescription":
                trigger = False
                for i, (text, score) in enumerate(rec_res):
                    trigger = True if "合计" in text or trigger == True else False
                        
                    text = MedicineDatabase.curr_false(text,0.6)
                    rec_res[i] = (text, score)
                    if text is not None:
                        dt_boxes_res.append(dt_boxes[i])
                        prescription_res.append(text)
                return dt_boxes_res,prescription_res,trigger
            else:
                return dt_boxes, rec_res
        else:
            rec_res, predict_time = text_sys.text_recognizer([img])
            rec_res_reu=MedicineDatabase.curr_false(rec_res[0][0], 0.4)
            return rec_res_reu

    