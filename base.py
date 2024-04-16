import os
import time
import signal
import multiprocessing


class MyProcess(multiprocessing.Process):
    def __init__(self, event) -> None:
        super(MyProcess, self).__init__()
        # signal.signal(signal.SIGUSR1, self.execute)

        self.event = event
        
    def run(self):
        while True:
            signal.signal(signal.SIGUSR1, self.execute)
            signal.pause()

    def execute(self, signum, frame) -> None:
        print("Custom function executed by PID:", os.getpid())
        self.event.set()

    def release(self):
        self.terminate()

if __name__ == "__main__":
    # 创建进程池并启动子进程
    num_processes = 3  # 设置子进程数量
    event = multiprocessing.Event()
    processes = [MyProcess(event) for _ in range(num_processes)]
    for p in processes:
        p.start()
    
    time.sleep(1)
    
    # 向所有子进程发送自定义信号（SIGUSR1）
    for p in processes:
        os.kill(p.pid, signal.SIGUSR1)
    
    for p in processes:
        event.wait()

    time.sleep(1)
    
    for p in processes:
        os.kill(p.pid, signal.SIGUSR1)
    
    for p in processes:
        p.release()