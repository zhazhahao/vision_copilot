import subprocess
import time
import traceback

from configs.video.audio_config import *
from utils.video.generate_fake_audio import generate_audio


from multiprocessing import shared_memory


# 生成并发送实时音频流到 RTSP 服务器
def send_realtime_audio_to_rtsp(ffmpeg_command,flag:mp.Value,exit_flag:mp.Value):
    try:
        ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)
        shared_audio_data = shared_memory.SharedMemory(create=True, size=176401)  # 创建共享内存
        audio_data = generate_audio(durations,sample_rate,0,0)    
        while exit_flag:
            a = time.time()
            if flag.value == b'\x00':
                audio_data = generate_audio(durations, sample_rate, 0,deleash=0)
            else:
                audio_data = generate_audio(durations, sample_rate, frequency,deleash=1)
                flag.value = b'\x00'
            ffmpeg_process.stdin.write(audio_data)
            if time.time() - a < 0.5:
                print(1 - time.time() + a)
            print(flag.value)
            ffmpeg_process.stdin.flush()
                # print(audio_data)
    
    except Exception:
        traceback.print_exc()
    except KeyboardInterrupt:
        print("Exited KeyBoardInterruption")
    ffmpeg_process.kill()
    shared_audio_data.unlink()
if __name__ == "__main__":
    process = mp.Process(target=send_realtime_audio_to_rtsp, args=(ffmpeg_audio_command,mp.Value("c", 0),mp.Value("i", 0)))
    process.start()
    process.join()