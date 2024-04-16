from multiprocessing import shared_memory
import os
from utils.video.send_audio_to_rtsp import load_ffmpeg_command_from_yaml
import multiprocessing as mp

try_num = 3000
resolution = [1920,1080,3]
shm_name = "shared_image_memory"
shared_memory_size = 1000000
rtsp_client_url = "rtsp://192.168.8.100/live"
rtsp_server_path = "/home/portable-00/VisionCopilot/pharmacy/dependency/rtsp_easy_server"
ffmpeg_video_command = load_ffmpeg_command_from_yaml('/home/portable-00/VisionCopilot/pharmacy/configs/video/configv1.yml', rtsp_client_url ,'ffmpeg_video_command')
