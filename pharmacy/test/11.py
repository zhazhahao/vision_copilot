li = ["要", "不要", "再考虑下", "要", "不要", "要"]
print(f"列表中出现次数最多的元素是：{max(li, key=li.count)} ，总出现次数：{li.count(max(li, key=li.count))}")
