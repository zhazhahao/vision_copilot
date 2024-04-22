import subprocess
import time
import traceback

from utils.videos import remove_nonblock
from utils.videos import generate_audio
from qinglang.utils.utils import Config
import multiprocessing as mp
from multiprocessing import shared_memory

# 生成并发送实时音频流到 RTSP 服务器
def send_realtime_audio_to_rtsp(ffmpeg_command,flag:mp.Value,exit_flag:mp.Value):
    try:
        audio_config = Config("configs/video/audio_config.yml")
        ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE,bufsize=0)
        shared_audio_data = shared_memory.SharedMemory(create=True, size=176401)  # 创建共享内存
        first_time = True
        while exit_flag:
            if first_time:
                ffmpeg_process.stdin.write(generate_audio(audio_config.durations,1,audio_config.frequency,0,32))
                ffmpeg_process.stdin.write(generate_audio(audio_config.durations,audio_config.sample_rate,audio_config.frequency,deleash=0))
                first_time = False
                continue
            if flag.value == b'\x00':
                continue
            else:
                print("While")
                audio_data = generate_audio(audio_config.durations,audio_config.sample_rate,audio_config.frequency,deleash=1) 
            ffmpeg_process.stdin.write(audio_data)
            ffmpeg_process.stdin.flush()
            flag.value = b'\x00'
    except Exception:
        traceback.print_exc()
    except KeyboardInterrupt:
        print("Exited KeyBoardInterruption")
    remove_nonblock(ffmpeg_process.stdin.fileno())
    ffmpeg_process.kill()
    shared_audio_data.unlink()
# if __name__ == "__main__":
#     rtsp_server_url = "rtsp://192.168.8.138:8554/aac"
# 
#     # Code for loader
#     ffmpeg_audio_command = load_ffmpeg_command_from_yaml('/home/portable-00/VisionCopilot/pharmacy/configs/video/configv1.yml', rtsp_server_url ,'ffmpeg_audio_command')
# 
#     process = mp.Process(target=send_realtime_audio_to_rtsp, args=(ffmpeg_audio_command,mp.Value("c", 0),mp.Value("i", 0)))
#     process.start()
#     process.join()