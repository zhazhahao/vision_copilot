
from utils.yolv_infer.curr_false import curr_false

def procession(img, text_sys, data_lists, options="process"):
    prescription_res = []
    dt_boxes, rec_res = text_sys(img)
    if options == "prescription":
        for i, (text, score) in enumerate(rec_res):
            text = curr_false(text, data_lists)
            rec_res[i] = (text, score)
            prescription_res.append(text)
        return prescription_res
    else:
        return dt_boxes,rec_res
