
import fcntl
import os
import subprocess


def achieve_process(ffmpeg_video_command,is_non_block=True):
    ffmpeg_process = subprocess.Popen(ffmpeg_video_command, stdout=subprocess.PIPE)
    if is_non_block:
        fd = ffmpeg_process.stdout.fileno()
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
    return ffmpeg_process