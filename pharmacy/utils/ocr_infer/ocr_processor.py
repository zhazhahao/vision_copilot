import torch
from utils.yolv_infer.curr_false import curr_false


def procession(img, text_sys, data_lists, options="process"):
    prescription_res = []
    keywords = []
    with torch.no_grad():
        dt_boxes, rec_res = text_sys(img)
        if options == "prescription":
            for i, (text, score) in enumerate(rec_res):
                text = curr_false(text, data_lists,0.6)
                rec_res[i] = (text, score)
                if text is not None:
                    prescription_res.append(text)
            return prescription_res
        else:
            return dt_boxes, rec_res
