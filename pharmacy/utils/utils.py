from qinglang.utils.utils import ClassDict, load_json
import re
import Levenshtein
from utils.ocr_infer.load_data_list import load_txt
from utils.yolv_infer.index_transfer import IndexTransfer


class MedicineDatabase:
    def __init__(self) -> None:
        self.data = load_json('database/medicine_database_en.json')
        self.data_lists = load_txt(self.source.medicine_names)
        self.index_transfer = IndexTransfer()

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, key: int) -> ClassDict:
        return self.data[key - 1]

    def find(self, name: str, **kwargs) -> int:
        assert all(key in self.data[0] for key in kwargs)
        
        candidates = [medicine for medicine in self.data if medicine['Name'] == name]
        
        for key, value in kwargs.items():
            candidates = [medicine for medicine in candidates if medicine[key] == value]
            
        return candidates

    def curr_false(self, text, max_threshold=0.4):
        search_ch_text = re.compile(r'[\u4e00-\u9fff]')
        search_brackets_text = re.compile(r'[()\u3000\uFF08\uFF09\u3010\u3011]')
        # search_num_text = re.compile(r'\d')
        if search_ch_text.search(text):
            if text in self.data_lists:
                return text
            # while searching number，you should do some improvements.
            text = re.sub(r'\d+[^0-9]*$', '', text)
            filtered_data_lists = self.data_lists.copy()
            if search_brackets_text.search(text):
                filtered_data_lists = [item for item in filtered_data_lists if search_brackets_text.search(item[0])]
            similarities = [Levenshtein.ratio(text, str2[0]) for str2 in filtered_data_lists]
            max_similarity = max(similarities)
            max_index = similarities.index(max_similarity)
            most_similar_drug = filtered_data_lists[max_index][0]
            if max_similarity > max_threshold:
                return most_similar_drug
            
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
    
    def tensor_converter(tensor):
        tensor = tensor.cpu()
        numpy_array = tensor.numpy()
        values_list = numpy_array.tolist()
        return values_list
    
    def drug_match(self, medicine_cls, prescription):
        med_name = self.index_transfer.get_drug_by_index(medicine_cls)
        if med_name["药品名称"] in prescription:
            return True
        else:
            return False