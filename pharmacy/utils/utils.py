from qinglang.utils.utils import ClassDict, load_json


class MedicineDatabase:
    def __init__(self) -> None:
        self.data = load_json('database/medicine_database_en.json')

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