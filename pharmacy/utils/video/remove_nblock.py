import fcntl
import os
def remove_nonblock(fd):
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    # 移除 O_NONBLOCK 标志位
    flags &= ~os.O_NONBLOCK
    fcntl.fcntl(fd, fcntl.F_SETFL, flags)