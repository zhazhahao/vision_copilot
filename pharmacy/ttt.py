import re

# 药品列表
medicines = [
    "脂肪乳氨基酸(17)葡萄糖(11%)注 1440ml/袋",
    "醋酸阿托西班注射液 ▲6.75mg0.9ml/支",
    "富露施（吸入用乙半胱氨）0.3g3ml/支",
    "利奈唑胺葡萄糖注射液 0.6g300ml/袋",
    "斯沃(利奈唑胺葡萄糖注射液) 0.6g300ml/袋",
    # 添加更多药品...
]

pattern = r"(.+?)\s*(▲?\d+(\.\d+)?[a-zA-Z%μ]*\d*(\.\d+)?(\/\d+)?)(ml\/\S+)"
text = "醋酸阿托西班注射液 ▲6.75mg0.9ml/支"

matches = re.findall(pattern, text)
if matches:
    for match in matches:
        print("药品名称:", match[0])
        print("剂量:", match[1])
else:
    print("未能匹配:", text)
