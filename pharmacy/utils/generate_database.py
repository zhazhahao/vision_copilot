import json
import pandas as pd

def find_adjacent_medicines(medicines, current_index, location_info):
    # 查找相邻药品
    adjacent_medicines = []
    for i, med in enumerate(medicines):
        if i != current_index:
            med_location_info = {
                '货架号': med.get('货架号', ''),
                '组': med.get('组', ''),
                '行': med.get('行', ''),
                '列': med.get('列', ''),
            }

            # 仅在药品拥有货架号、行、列、组等属性时再进行比较
            if all(med_location_info.values()):
                if (
                    med_location_info['货架号'] == location_info['货架号'] and
                    med_location_info['组'] == location_info['组'] and
                    (med_location_info['行'] == location_info['行'] or med_location_info['列'] == location_info['列'])
                ):
                    adjacent_medicines.append(med)

    return adjacent_medicines


def add_adjacent_medicine_info(medicines):
    # 为每个药品添加相邻药品信息
    for i, medicine in enumerate(medicines):
        location_info = {
            '货架号': medicine.get('货架号', ''),
            '组': medicine.get('组', ''),
            '行': medicine.get('行', ''),
            '列': medicine.get('列', ''),
        }

        # 仅在药品拥有货架号、行、列、组等属性时才查找相邻药品
        if all(location_info.values()):
            # 查找相邻药品
            adjacent_medicines = find_adjacent_medicines(medicines, i, location_info)
            if adjacent_medicines:
                medicine['相邻药品信息'] = [adj_med['药品名称'] for adj_med in adjacent_medicines]

    return medicines

# 读取 Excel 文件
df = pd.read_excel('/home/portable-00/VisionCopilot/pharmacy/database/注射剂目录2024.4.27.xlsx')

# 将货位号拆分成货架号、行、组、列
if df['货位号'] is not None:
    df['货架号'] = df['货位号'].str[0]
    df['组'] = df['货位号'].str[1:3]
    df['行'] = df['货位号'].str[3:5]
    df['列'] = df['货位号'].str[5:]
else:
    df['货架号'] = ''
    df['组'] = ''
    df['行'] = ''
    df['列'] = ''
# 将 DataFrame 转换成 JSON 格式
json_data = df.to_json(orient='records', force_ascii=False)

# 将 JSON 数据写入文件
with open('/home/portable-00/VisionCopilot/pharmacy/database/medicine_database_tmp.json', 'w', encoding='utf-8') as f:
    f.write(json_data)
print("JSON 文件已生成。")


file_path = "database/medicine_database_tmp.json"
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# 遍历每个药品，为其添加索引属性
for index, item in enumerate(data, start=1):
    item["index"] = index

final_data = add_adjacent_medicine_info(data)

# 将更新后的数据写回原始JSON文件
with open(file_path, "w", encoding="utf-8") as file:
    json.dump(final_data, file, indent=2, ensure_ascii=False)
print("索引已成功添加到原始JSON文件中。")