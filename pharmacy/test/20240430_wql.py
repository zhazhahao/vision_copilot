
import numpy as np
from itertools import groupby
# from qinglang.utils.utils import timer
# # 测试代码
# lst = [0, 0, 0, 'A', 'A', 'A', 'A', 'A', 0, 0, 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C'] * 30 * 60 * 5

# @timer(1000)
# def group_consecutive_elements_np(lst):
#     lst_np = np.array(lst)
#     changes = np.where(lst_np[:-1] != lst_np[1:])[0] + 1
#     groups = [(lst_np[start], np.arange(start, end)) for start, end in zip(np.hstack((0, changes)), np.hstack((changes, len(lst))))]
#     return groups

# @timer(1000)
# def group_consecutive_elements_py(lst):
#     groups = []
#     start_index = 0
#     for key, group in groupby(lst):
#         group_length = len(list(group))
#         groups.append((key, start_index + np.arange(group_length)))
#         start_index += group_length
#     return groups

# # 测试代码
# result = group_consecutive_elements_np(lst)

# # for idx, (elem, indices) in enumerate(result):
# #     print("第{}次：({}, {})".format(idx + 1, elem, indices))
    
# result = group_consecutive_elements_py(lst)

# # for idx, (elem, indices) in enumerate(result):
# #     print("第{}次：({}, {})".format(idx + 1, elem, indices))

# import numpy as np

# # 创建一个 NumPy 数组
# arr = np.array([1, 2, 3, 4, 5])

# # 定义指定列表
# lst = [2, 4]

# # 使用 numpy.in1d 函数创建一个布尔数组，表示数组中的每个元素是否在指定列表中
# not_in_lst_indices = np.in1d(arr, lst, invert=True)

# # 使用 count_nonzero 函数统计布尔索引中为 True 的数量
# count = np.count_nonzero(not_in_lst_indices)

# print("不在列表 {} 中的元素数量为: {}".format(lst, count))
# print(len([key for key in arr if key not in lst]))


from qinglang.utils.utils import load_json
from pprint import pprint

def group_consecutive_elements(lst):
    groups = []
    current_index = 0
    for key, group in groupby(lst):
        group_length = len(list(group))
        # groups.append((key, current_index + np.arange(group_length)))
        groups.append((key, [current_index, current_index + group_length - 1]))
        current_index += group_length
    return groups

anno = load_json("/home/portable-00/data/20240313_160556/catch_action.json")
group = group_consecutive_elements(anno)
pprint(group)