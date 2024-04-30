import time
import cv2
from modules.camera_processor import CameraProcessor

if __name__ == "__main__":
    camera = CameraProcessor()
    camera.start()
    i = 0
    a = time.time()
    while(True):
        try:
            bools,mat = camera.achieve_image()
            scale_percent = 50  # 设置缩放比例
            width = int(mat.shape[1] * scale_percent / 200)
            height = int(mat.shape[0] * scale_percent / 200)
            dim = (width, height)
            resized_image = cv2.resize(mat, dim, interpolation=cv2.INTER_AREA)
            if bools:
                # cv2.imwrite("/home/portable-00/VisionCopilot/pharmacy/images/"+str(i)+".jpg",mat)
                # cv2.imshow("win_name",resized_image)
                i = i+1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        except KeyboardInterrupt:
            print("KeyBoardInterruption")
            break
        except Exception as e:
            continue
    camera.end_process()
    
import re

# 定义可能出现在药品名称前面的关键词列表，包括所有数字
before_keywords = ["集采", "预灌封", "预充式", "预充装","小" ,r"\[N-R\]", r"\[N-N\]", r"\[30R\]", r"\[50R\]"
                   , r"笔芯\[Y-R\]", r"\(笔芯\)",r"\(水果味\)", r"笔芯",r"预充",r"预装式",r"预冲式",r"预填充"]
unit_keywords = ["单位", "万单位", "ml", "mg", "g", "IU", "WU" ,"UG", "mgI","ML","喷","微克","吸","粒","μg","揿","揿×","mg:","g:","喷\*"]
keyword_pattern = "|".join(before_keywords)
unit_pattern = "|".join(unit_keywords)
pattern = rf"^(.+?)\s*(?:{keyword_pattern})?\s*(((\d+(\.\d+)?)%?[a-zA-Z%μ]*\s*)+({unit_pattern})((\d+(\.\d+)?\s*)+({unit_pattern})\s*)*x?\d*\s*)/\s*(\S+)$"

with open('/home/portable-00/VisionCopilot/pharmacy/database/medicine_names.txt', 'r', encoding='utf-8') as file:
    for line in file:
        text = line.strip()
        match = re.search(pattern, text)
        if match:
            print("药品名称:", match.group(1))
            print("剂量:", text[match.end(1):])
        else:
            print("未能匹配:", text)