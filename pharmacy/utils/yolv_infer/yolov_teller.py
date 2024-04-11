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