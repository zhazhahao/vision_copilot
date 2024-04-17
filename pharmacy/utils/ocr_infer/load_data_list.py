def load_txt(path):
    data_list=[]
    with open(path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    for line in lines:
        # 去除行尾的换行符并按空格切分
        items = line.strip().split()
        data_list.append(items)
        # 使用第一列元素作为键，整行作为对应的值
    return data_list