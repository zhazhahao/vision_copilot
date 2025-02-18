import re
import time
import torch
from utils.yolv_infer.curr_false import curr_false

def procession(img, text_sys, data_lists, options="process"):
    prescription_res = []
    dt_boxes_res = []
    keywords = []
    with torch.no_grad():
        if options != "Single":
            dt_boxes, rec_res = text_sys(img)
            # print(rec_res)
            call_box = []
            if options == "prescription":
                trigger = False
                for i, (text, score) in enumerate(rec_res):
                    trigger = True if "合计" in text or trigger == True else False
                    # 构建正则表达式模式
                    regex = re.compile("|".join(["/"]))
                    # 在文本中查找匹配的关键字
                    matches = regex.search(text)
                    # print(text)
                    if matches and score >= 0.8:
                        text.replace(" ", "")
                        call_box.append(text)
                    text = curr_false(text, data_lists,0.6)
                    rec_res[i] = (text, score)
                    if text is not None:
                        dt_boxes_res.append(dt_boxes[i])
                        prescription_res.append(text)
                return dt_boxes_res,prescription_res,trigger,call_box
            else:
                return dt_boxes, rec_res
        else:
            rec_res, predict_time = text_sys.text_recognizer([img])
            rec_res=curr_false(rec_res[0][0], data_lists, 0.4)
            return rec_res
