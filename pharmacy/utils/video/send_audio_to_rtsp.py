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

def send_audio_to_rtsp(ffmpeg_command, shared_audio_data:shared_memory.SharedMemory, exit_flag):

    ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)
    time.sleep(2)
    while not exit_flag.value:
        try:
            res = copy.deepcopy(shared_audio_data.buf.tobytes())
            ffmpeg_process.stdin.write(res[:176400])
            ffmpeg_process.stdin.flush()
        except IOError:
            traceback.print_exc()
            continue
        except KeyboardInterrupt:
            break
        except SystemExit:
            print("Worker received SIGTERM")
            sys.exit()
    ffmpeg_process.stdin.flush()
    remove_nonblock(ffmpeg_process.stdin.fileno())
    ffmpeg_process.terminate()

    
def load_ffmpeg_command_from_yaml(yaml_file, rtsp_url, config_name):
    with open(yaml_file, 'r') as file:
        config_text = file.read()
        template = Template(config_text)
        config_text_rendered = template.render(rtsp_url_placeholder=rtsp_url)
        config = yaml.safe_load(config_text_rendered)
        
    ffmpeg_command = ['-' if x == None else x for x in config[config_name]]
    return ffmpeg_command


