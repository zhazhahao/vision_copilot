# 发送音频数据到 RTSP 服务器
import copy
from multiprocessing import shared_memory
import subprocess
import sys
import time
import traceback
import yaml
from jinja2 import Template

from utils.video.remove_nblock import remove_nonblock



    
def load_ffmpeg_command_from_yaml(yaml_file, rtsp_url, config_name):
    with open(yaml_file, 'r') as file:
        config_text = file.read()
        template = Template(config_text)
        config_text_rendered = template.render(rtsp_url_placeholder=rtsp_url)
        config = yaml.safe_load(config_text_rendered)
        
    ffmpeg_command = ['-' if x == None else x for x in config[config_name]]
    return ffmpeg_command


