import copy
import fcntl
import os
import subprocess
import time
import traceback
import cv2
import numpy as np
from functools import reduce
import multiprocessing as mp


from send_video.instance.audio_sender import send_realtime_audio_to_rtsp,ffmpeg_audio_command

from utils.jpeg_vailder import is_jpeg_header,is_jpeg_end,has_jpeg_end

from configs.video_config import *
from configs.achieve_process import achieve_process

def image_put(exit_flag):
    # 启动ffmpeg进程
    ffmpeg_process = achieve_process(ffmpeg_video_command)
    try:
        try_num = 0
        frame = 0
        back_image = b''
        while not exit_flag.value:
            # 读取单帧图像数据
            a = time.time()
            raw_image = ffmpeg_process.stdout.read(100000)  # 假设单帧图像大小不超过 65536 字节
            if try_num >= 3000 and raw_image is not None and len(raw_image) == 0:
                try_num = 0
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
                    shm.buf[:5] = size_bytes  # 将图像大小作为前五个字节存入共享内存
                    shm.buf[5:5+len(raw_image)] = raw_image  # 存入完整的 raw image 数据
                    print("Answer" + str(len(raw_image)))
                    frame += 1
                    print(time.time() - a)
                else:
                    if len(back_image) == 0:
                        back_image = back_image + raw_image
                    else:
                        png_end = has_jpeg_end(raw_image)
                        if png_end[0]:
                            res_image = back_image + raw_image[:png_end[1]]
                            back_image = raw_image[png_end[1]+1:]
                            size_bytes = len(res_image).to_bytes(5, byteorder='big')
                            shm.buf[:5] = size_bytes  # 将图像大小作为前五个字节存入共享内存
                            shm.buf[5:5+len(res_image)] = res_image  # 存入完整的 raw image 数据
                            print("Late Answer" +str(len(res_image)))
                            print(time.time() - a)
                        else:
                            back_image = back_image + raw_image
                try_num = 0
            else:
                try_num += 1
            # 显示图像
        exit_flag.value = 0
    except Exception as e:
        traceback.print_exc()
        pass
    except KeyboardInterrupt:
        pass    
    shm.close()
    shm.unlink()
    ffmpeg_process.terminate()

def image_get(camera_name, exit_flag,infor_value):
    try:
        while not exit_flag.value:
            a = time.time()
            shm_now = copy.deepcopy(shm.buf.tobytes())
            # 从共享内存中读取图像大小（前五个字节）
            image_size = int.from_bytes(shm_now[:5], byteorder='big')
            if image_size == 0 or image_size >= 10000000:
                print("No")
                time.sleep(0.1)
                continue
            # 根据图像大小从共享内存中读取图像数据
            binary_data = shm_now[5:5+image_size]
            image = cv2.imdecode(np.frombuffer(binary_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                image_array = np.frombuffer(image, dtype=np.uint8)
            else:
                continue
            if (False):# fill in logic there
                infor_value |= infor_value
            # 将图像数据转换为图像数组
            if image_array is not None and image_array.shape[0] == reduce((lambda x, y: x * y), resolution):
                image = image_array.reshape(tuple(resolution[1::-1] + resolution[2:]))
                cv2.imshow(camera_name, image)
            # 等待键盘输入，如果按下 'q' 键则退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            print("process_time" + str(time.time() - a))
        # 关闭所有窗口
        cv2.destroyAllWindows()
    
    except Exception as e:
        traceback.print_exc()

        
def run_single_camera():
    camera_name = "FinalVersion"
    exit_flag = mp.Value("i", 0)  # 退出标志，0表示不退出，1表示退出
    infor_value = mp.Value("c", 0)
    processes = [mp.Process(target=image_put, args=(exit_flag,)),
                 mp.Process(target=image_get, args=(camera_name, exit_flag,infor_value)),
                 mp.Process(target=send_realtime_audio_to_rtsp,args=(ffmpeg_audio_command,infor_value,exit_flag))
                 ]
    for process in processes[:-1]:
        process.daemon = True  # 防止主进程挂掉子进程变成僵尸进程
        process.start()
    processes[-1].start()
    try:
        [process.join() for process in processes]
    except KeyboardInterrupt:
        print("接收到 KeyboardInterrupt")
        exit_flag.value = 1

    for process in processes:
        process.terminate()
    print("主进程退出")
run_single_camera()