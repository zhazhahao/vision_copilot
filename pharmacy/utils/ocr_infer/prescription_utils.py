from collections import Counter
from qinglang.utils.utils import ClassDict



class PreScriptionRecursiveObject:
    def __init__(self) -> None:
        self.from_index = 0
        self.to_index = 0 
        self.times = 0
        self.recursive_epoch = 3
        self.recursive_obj = []
        self.static_obj = []
        self.quest_refer = {}
        self.quest_loss = []
    def update(self, list,times) -> None:
        self.recursive_obj = self._merge_drug_lists(self.recursive_obj,list, times)
        
    def achieve(self):
        return self.static_obj.extend(self.recursive_obj)
    
    def _merge_drug_lists(self, list1, list2, times):
        checked_list_time = Counter()
        merged_list = []
        reserve_list = []
        i,j = 0,0
        first_there = True
        if len(list1) == 0:
            return list2
        if len(list1) > 1 and len(list2) == 1:
            return list1
        while i < len(list1) and j < len(list2):
            if list1[i] == list2[j]:
                self.from_index = i if first_there else self.from_index
                merged_list.append(list1[i]) 
                first_there = False
                i += 1
                j += 1
            elif list2[j] in list1 and list1.index(list2[j]) >= i:
                # if list1.index(list2[j]) == i + 1:
                #     merged_list.extend(reserve_list)
                reserve_list = []
                merged_list.append(list1[i])
                i += 1
            else:
                self.from_index = i if first_there else self.from_index
                first_there = False
                reserve_list.append(list2[j])
                j += 1
        merged_list.extend(list1[i:])
        merged_list.extend(list2[j:])
        self.to_index = merged_list.__len__()
        if len(reserve_list) > len(list2) - len(reserve_list) and (self.static_obj.__len__() == 0 or times - self.static_obj[self.static_obj.__len__() - 1][0] > 10):
            # print(times)
            self.static_obj.append([times,merged_list])
            merged_list = []
        merged_list.extend(reserve_list)
        # for item in reversed(merged_list):
        #     result_list.insert(0, item) if item not in result_list else None
        return merged_list
    
    def _check_merge_drugs(self,image):
        if self.static_obj.__len__():
            for i in range(len(self.static_obj)):
                if self.static_obj[i][0] >= image[0]:
                    break
            # j = 0
            # print(self.static_obj[i][1],image[1],self.find_positions(self.static_obj[i][1],
            #                           image[1]))
            
    def find_positions(self,arr1, arr2):
        positions = []
        used_indexes = set()  # 记录已经使用过的索引
        for item in arr1:
            found = False
            for i, x in enumerate(arr2):
                if x == item and i not in used_indexes:
                    positions.append(i)
                    used_indexes.add(i)
                    found = True
                    break
            if not found:
                positions.append(None)
        return positions                                       

class FrameMaxMatchingCollections(ClassDict):
    def __init__(self, *args, **kwargs) -> None:
        self.result_counter = {}
        
    def update(self ,frame, max_candicated, times):
        for result in max_candicated[1]:
            self.result_counter[result] = ClassDict(
                    tickles = times,
                    max_candicated = max_candicated,
                    res_frame = frame,
                    counts = 1 if result not in self.result_counter.keys() else self.result_counter[result].counts + 1
                ) if result not in self.result_counter or max_candicated[1].__len__() > self.result_counter[result].max_candicated[1].__len__() else ClassDict(
                    tickles = self.result_counter[result].tickles,
                    max_candicated = self.result_counter[result].max_candicated,
                    res_frame = self.result_counter[result].res_frame,
                    counts = self.result_counter[result].counts + 1
                )
    def values(self):
        return {elements.tickles: ClassDict(frame=elements.res_frame, max_candicated=elements.max_candicated) 
                              for i, elements in self.result_counter.items()}