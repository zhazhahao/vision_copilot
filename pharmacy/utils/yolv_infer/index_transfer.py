class IndexTransfer:
    def __init__(self) -> None:
        pass

    def cls2name(self, drug_list: list, index: int = 0):
        if index < 0:
            return None
        try:
            return drug_list[index][0]
        except (IOError, IndexError):
            return None
        
        #根据药品名称返回med迭代器
    def find_medicine_from_json_by_name(self, medicines, target_name): 
        for med in medicines:
            if med.get('药品名称') == target_name:
                return med
            
    def name2cls(self, drug_list: list,name: str=''):
            # 确保索引从0开始
        matching_indices = []
        if name == None:
            return None
        try:
            for i in range(len(drug_list)):
                if name == drug_list[i][0]:
                    matching_indices.append(i - 1)
        except (IOError, IndexError):
            # 如果文件不存在或索引超出范围
            return None
        return matching_indices