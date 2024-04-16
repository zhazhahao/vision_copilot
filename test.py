import multiprocessing
import signal
import os
import time

class MyProcess(multiprocessing.Process):
    def __init__(self):
        super(MyProcess, self).__init__()
        
    def custom_function(self, signum, frame):
        time.sleep(3)
        print("Custom function executed by PID from child process:", os.getpid())

    def run(self):
        signal.signal(signal.SIGUSR1, self.custom_function)
        # print("Child process PID:", os.getpid())
        while True:
            time.sleep(1)  # 为了防止子进程退出，这里添加了一个睡眠

if __name__ == "__main__":
    # 创建子进程但不启动
    p = MyProcess()

    # 启动子进程
    p.start()
    
    # 等待子进程启动
    time.sleep(1)
    
    # 向子进程发送自定义信号（SIGUSR1）
    os.kill(p.pid, signal.SIGUSR1)
    print('from main process')
    
    # # 等待用户输入并检查是否为 'q'
    # user_input = ''
    # while True:
    #     user_input = input("Press 'q' to exit: ")
    #     if user_input.lower() == 'q':
    #         p.terminate()
    #         break
    #     os.kill(p.pid, signal.SIGUSR1)
