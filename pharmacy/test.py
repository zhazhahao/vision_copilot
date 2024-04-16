import time

# 输出进度条
def print_progress_bar(iteration, total, prefix='', suffix='', length=30, fill='█'):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    # 如果达到100%，则换行
    if iteration == total:
        print()

# 模拟进度更新
total = 100
for i in range(1, total + 1):
    time.sleep(0.1)  # 模拟一些工作
    print_progress_bar(i, total, prefix='Progress:', suffix='Complete', length=50)

print("Task completed!")
