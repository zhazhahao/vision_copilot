def find_positions(arr1, arr2):
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


arr1 = [1, 2, 3, 4]
arr2 = [4, 4, 2, 1]
print("元素位置：", find_positions(arr1, arr2))
