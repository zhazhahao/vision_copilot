def rearrange_list(lst):
    seen = set()  # 用集合来跟踪已经遍历过的元素
    unique = []   # 存放未曾遍历过的元素
    for item in lst:
        if item not in seen:  # 如果元素未曾遍历过
            unique.append(item)  # 将元素添加到未曾遍历过的元素列表中
            seen.add(item)  # 将元素加入集合中
        else:
            unique.remove(item)  # 如果元素已经遍历过，则先将其从未曾遍历过的列表中移除
            unique.append(item)  # 然后将其添加到未曾遍历过的列表的末尾
    return unique


# 示例用法
medications = ['呋塞米注射液', '氯化钙注射液', '肝素钠封管注射液', '注射用亚叶酸钙', '瑞白(人粒细胞刺激因子注射液)', 
               '呋塞米注射液', '氯化钙注射液', '善宁(醋酸奥曲肽注射液)', '肝素钠封管注射液', '注射用亚叶酸钙', 
               '浓氯化钠注射液', '氯化钠注射液[塑料安瓿]', '肝素钠封管注射液', '注射用亚叶酸钙', '肝素钠封管注射液', 
               '注射用亚叶酸钙', '肝素钠封管注射液', '注射用亚叶酸钙', '肝素钠封管注射液', '注射用亚叶酸钙', 
               '肝素钠封管注射液', '注射用亚叶酸钙', '氯化钾注射液[塑料安瓿]', '氯化钙注射液']

print("原始列表:", medications)
rearranged_list = rearrange_list(medications)
print("操作后的列表:", rearranged_list)
