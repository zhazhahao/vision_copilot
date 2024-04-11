from multiprocessing import shared_memory
import os
from utils.video.send_audio_to_rtsp import load_ffmpeg_command_from_yaml
import multiprocessing as mp

try_num = 3000
resolution = [1280,720,3]
shm_name = "shared_image_memory"
shared_memory_size = 1000000
rtsp_client_url = "rtsp://192.168.3.100/live"
ffmpeg_video_command = load_ffmpeg_command_from_yaml('pharmacy/configs/video/configv1.yml', rtsp_client_url ,'ffmpeg_video_command')
