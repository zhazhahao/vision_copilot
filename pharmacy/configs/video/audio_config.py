import multiprocessing as mp
from utils.video.send_audio_to_rtsp import load_ffmpeg_command_from_yaml

# Code for audioSender
sample_rate = 44100  # sampling r
frequency = 440  # frequency 
lock = mp.Lock()
rtsp_server_url = "rtsp://192.168.3.79:8554/aac"
durations = 2

# Code for loader
ffmpeg_audio_command = load_ffmpeg_command_from_yaml('/home/portable-00/VisionCopilot/pharmacy/configs/video/configv1.yml', rtsp_server_url ,'ffmpeg_audio_command')
