import copy
from functools import reduce
from multiprocessing import shared_memory
import os
import subprocess
import traceback

import cv2
import numpy as np
import multiprocessing as mp

from utils.videos import achieve_process, load_ffmpeg_command_from_yaml
from modules.audio_sender import send_realtime_audio_to_rtsp
from utils.videos import remove_nonblock
from utils.videos import is_jpeg_header,is_jpeg_end,has_jpeg_end
from qinglang.utils.utils import Config

class CameraProcessor:
    def __init__(self,doaudio=True,rtsp_client_path="rtsp://192.168.8.100/live"):
        self.camera_name = "DRIFTX3"
        self.rtsp_client_path = rtsp_client_path
        self.doaudio = doaudio
        self.exit_flag = mp.Value("i", 0)  # Exit Mark
        self.infor_value = mp.Value("c", 0) # Audio Mark
        self.server_process = subprocess.Popen("./mediamtx",cwd="/home/portable-00/VisionCopilot/pharmacy/dependency/rtsp_easy_server")
        self.config = Config("configs/video/video_config.yml")
        ffmpeg_audio_command = load_ffmpeg_command_from_yaml('/home/portable-00/VisionCopilot/pharmacy/configs/video/configv1.yml', self.config.rtsp_server_url ,'ffmpeg_audio_command')
        self.try_num = 0
        if doaudio:
            self.processes = [
                mp.Process(target=self._image_put_thread,args=(self.exit_flag,)),
                mp.Process(target=send_realtime_audio_to_rtsp,args=(ffmpeg_audio_command,self.infor_value,self.exit_flag))
            ]
        else:
            self.processes = [
                mp.Process(target=self._image_put_thread,args=(self.exit_flag,)),
                ]   
        if os.path.exists(f"/dev/shm/shared_image_memory"):
            os.remove(f"/dev/shm/shared_image_memory")
        self.shm = shared_memory.SharedMemory("shared_image_memory",create=True,size=self.config.shared_memory_size)
        self.start()
        
    def _image_put_thread(self,exit_flag):
        ffmpeg_video_command = load_ffmpeg_command_from_yaml('/home/portable-00/VisionCopilot/pharmacy/configs/video/configv1.yml', self.rtsp_client_path ,'ffmpeg_video_command')
        ffmpeg_process = achieve_process(ffmpeg_video_command)
        try:
            back_image = b''
            while not exit_flag.value:
                # 读取单帧图像数据
                # a = time.time()
                raw_image = ffmpeg_process.stdout.read(self.config.shared_memory_size)  # 假设单帧图像大小不超过 65536 字节
                if self.try_num >= 3000 and raw_image is not None and len(raw_image) == 0:
                    self.try_num = 0
                    ffmpeg_process.terminate()
                    ffmpeg_process = achieve_process(ffmpeg_video_command)
                    # ffmpeg_process = subprocess.Popen(ffmpeg_video_command, stdout=subprocess.PIPE)
                    # fd = ffmpeg_process.stdout.fileno()
                    # flags = fcntl.fcntl(fd, fcntl.F_GETFL)
                    # fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
                # 解码图像数据
                if raw_image is not None and len(raw_image) != 0:
                    if is_jpeg_header(raw_image) and is_jpeg_end(raw_image):
                        size_bytes = len(raw_image).to_bytes(5, byteorder='big')
                        self.shm.buf[:5] = size_bytes  # 将图像大小作为前五个字节存入共享内存
                        self.shm.buf[5:5+len(raw_image)] = raw_image  # 存入完整的 raw image 数据
                        # print("Answer" + str(len(raw_image)))
                        # frame += 1
                        # print(time.time() - a)
                    else:
                        if len(back_image) == 0:
                            back_image = back_image + raw_image
                        else:
                            png_end = has_jpeg_end(raw_image)
                            if png_end[0]:
                                res_image = back_image + raw_image[:png_end[1]]
                                back_image = raw_image[png_end[1]+1:]
                                size_bytes = len(res_image).to_bytes(5, byteorder='big')
                                self.shm.buf[:5] = size_bytes  # 将图像大小作为前五个字节存入共享内存
                                try:
                                    self.shm.buf[5:5+len(res_image)] = res_image  # 存入完整的 raw image 数据
                                except:
                                    print(type(res_image))
                                # print("Late Answer" +str(len(res_image)))
                                # print(time.time() - a)
                            else:
                                back_image = back_image + raw_image
                    self.try_num = 0
                else:
                    
                    # print(self.try_num)
                    self.try_num += 1
                # 显示图像
            exit_flag.value = 0
        except Exception as e:
            traceback.print_exc()
            pass
        except KeyboardInterrupt:
            pass    
        self.shm.unlink()
        remove_nonblock(ffmpeg_process.stdout.fileno())
        ffmpeg_process.kill()
        
    def start(self):
        try:
            if self.doaudio:
                for process in self.processes[:-1]:
                    process.daemon = True
                    process.start()
                self.processes[-1].start()
            else:
                self.processes[0].start()
        except Exception as e:
            print(e)
    
    def end_process(self):
        self.exit_flag.value |= self.exit_flag.value
        self.server_process.terminate()
        for process in self.processes:
            process.join()
        print("主进程退出")
    
    def achieve_image(self):
        shm_now = copy.deepcopy(self.shm.buf.tobytes())
        # 从共享内存中读取图像大小（前五个字节）
        image_size = int.from_bytes(shm_now[:5], byteorder='big')
        if image_size == 0 or image_size >= self.config.shared_memory_size * 10:
            # print(image_size)
            return False,None
        binary_data = shm_now[5:5+image_size]
        image = cv2.imdecode(np.frombuffer(binary_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is not None and image.size == reduce((lambda x, y: x * y), self.config.resolution):
            return True,image
        return False,None
    
    def send_wrong(self):
        # print("Got wrong medicine")
        self.infor_value.value  = 1
    
    def properity(self):
        return {"CameraName":self.camera_name,"Resolution":self.config.resolution}