def tensor_converter(tensor):
    tensor = tensor.cpu()
    numpy_array = tensor.numpy()
    values_list = numpy_array.tolist()
    return values_list

def find_medicine_by_name(medicines, target_name):
    for med in medicines:
        if med.get('药品名称') == target_name:
            return med

def get_drug_by_index(cls,data):
    for med in data:
        if med.get('index') == cls:
            return med
    return None

def get_drug_name_by_index(drug_list_path: str = '/home/portable-00/VisionCopilot/pharmacy/database/medicine_names.txt', index: int = 0):
    # 确保索引从0开始
    if index < 0:
        return None
    try:
        # 读取药品列表文件
        with open(drug_list_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            # 通过索引获取药品名称
            drug_name = lines[index].strip()
            return drug_name
    except (IOError, IndexError):
        # 如果文件不存在或索引超出范围
        return None
