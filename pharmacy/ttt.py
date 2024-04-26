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