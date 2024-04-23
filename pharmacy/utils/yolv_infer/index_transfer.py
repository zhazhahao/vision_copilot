from qinglang.utils.utils import Config, load_json
from utils.ocr_infer.load_data_list import load_txt



class IndexTransfer:
    def __init__(self) -> None:
        self.data = load_json(self.source.medicine_database)
        self.data_lists = load_txt(self.source.medicine_names)

    def cls2name(self, index: int = 0):
        if index < 0:
            return None
        try:
            return self.data_lists[index]
        except (IOError, IndexError):
            return None
        
        #根据药品名称返回med迭代器
    def find_medicine_from_json_by_name(self, target_name): 
        for med in self.data:
            if med.get('药品名称') == target_name:
                return med
            
    def name2cls(self, name: str=''):
            # 确保索引从0开始
        matching_indices = []
        if name == None:
            return None
        try:
            for i in range(len(self.data_lists)):
                if name == self.data_lists[i]:
                    matching_indices.append(i - 1)
        except (IOError, IndexError):
            # 如果文件不存在或索引超出范围
            return None
        return matching_indices
    
    def get_drug_by_index(self, cls):
        for med in self.data:
            if med.get('index') == cls:
                return med
        return None