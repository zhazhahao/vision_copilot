import traceback

from configs.video.audio_config import *
from utils.video.send_audio_to_rtsp import send_audio_to_rtsp
from utils.video.generate_fake_audio import generate_audio


from multiprocessing import shared_memory


# 生成并发送实时音频流到 RTSP 服务器
def send_realtime_audio_to_rtsp(ffmpeg_command,flag:mp.Value,exit_flag:mp.Value):
    try:
        shared_audio_data = shared_memory.SharedMemory(create=True, size=176401)  # 创建共享内存
        audio_data = generate_audio(durations,sample_rate,frequency,100)
        shared_audio_data.buf[:len(audio_data)] = audio_data         
        process=mp.Process(target=send_audio_to_rtsp, args=(ffmpeg_command, shared_audio_data,exit_flag))
        process.start()
        while exit_flag:
            audio_data = generate_audio(durations, sample_rate, frequency) if flag.value else generate_audio(durations, sample_rate, frequency,100)
            with lock:
                shared_audio_data.buf[:len(audio_data)] = audio_data
    except Exception:
        traceback.print_exc()
    except KeyboardInterrupt:
        print("Exited KeyBoardInterruption")
    shared_audio_data.unlink()
    process.terminate()
if __name__ == "__main__":
    process = mp.Process(target=send_realtime_audio_to_rtsp, args=(ffmpeg_audio_command,mp.Value("c", 0),mp.Value("i", 0)))
    process.start()
    process.join()