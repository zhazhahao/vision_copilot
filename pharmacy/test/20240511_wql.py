import sys
import os
import time
from qinglang.utils.utils import Config
import logging
config = Config('configs/hand_detection.yaml')
source = Config('configs/source.yaml')
from qinglang.utils.io import clear_lines

def clear_lines(n):
    # '\033[F' 是ANSI转义序列，用于将光标移动到上一行的开头
    for _ in range(n):
        sys.stdout.write('\033[F')  # 光标上移一行
        sys.stdout.write('\033[K')  # 清除当前行

from mmdeploy_runtime import Detector
clear_lines(3)
Detector.log = None
detector = Detector(model_path=source.onnx_path, device_name=config.device)

time.sleep(1)