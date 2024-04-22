import fcntl
import os
import subprocess
import subprocess
import yaml
from jinja2 import Template
import numpy as np


def achieve_process(ffmpeg_video_command,is_non_block=True):
    ffmpeg_process = subprocess.Popen(ffmpeg_video_command, stdout=subprocess.PIPE)
    if is_non_block:
        fd = ffmpeg_process.stdout.fileno()
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
    return ffmpeg_process

def generate_audio(duration, sample_rate = 44100, frequency = 440, deleash = 1, length = 32767):
    num_samples = int(duration * sample_rate)
    time = np.linspace(0, duration, num_samples)
    audio_data = np.sin(2 * np.pi * frequency * time)
    audio_data = (audio_data * deleash * length).astype(np.int16)  # 将数据缩放为16位整数
    return audio_data.tobytes()

def is_jpeg_header(image_data, header=b'\xFF\xD8\xFF'):
    # 检查图像数据的开头是否与 JPEG 头部匹配
    first_bytes = image_data[:len(header)]
    return first_bytes == header

def is_jpeg_end(image_data, end_bytes=b'\xFF\xD9'):
    # 检查图像数据的最后几个字节是否与 JPEG 结尾标记匹配
    last_bytes = image_data[-len(end_bytes):]
    return last_bytes == end_bytes

def has_jpeg_header(image_data, header=b'\xFF\xD8\xFF'):
    # 检查图像数据的开头是否与 JPEG 头部匹配
    start_pos = image_data.find(header)
    return start_pos != -1, start_pos + len(header)

def has_jpeg_end(image_data, end=b'\xFF\xD9'):
    # 从后往前搜索图像数据，找到是否存在 JPEG 结尾标识符
    end_pos = image_data.rfind(end)
    return end_pos != -1, end_pos + len(end)

def remove_nonblock(fd):
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    # 移除 O_NONBLOCK 标志位
    flags &= ~os.O_NONBLOCK
    fcntl.fcntl(fd, fcntl.F_SETFL, flags)  
# 发送音频数据到 RTSP 服务器
   
def load_ffmpeg_command_from_yaml(yaml_file, rtsp_url, config_name):
    with open(yaml_file, 'r') as file:
        config_text = file.read()
        template = Template(config_text)
        config_text_rendered = template.render(rtsp_url_placeholder=rtsp_url)
        config = yaml.safe_load(config_text_rendered)
        
    ffmpeg_command = ['-' if x == None else x for x in config[config_name]]
    return ffmpeg_command


