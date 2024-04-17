import os
import time
import cv2
from modules.yoloc_dector import YolovDector

class MainProcess:
    def __init__(self) -> None:
        self.max_opportunity = 10
        self.yoloc_decctor = YolovDector()

    def run(self):
    # this loop is for read each image in this foder,directory_name is the foder name with images.
        for filename in os.listdir(r"/home/portable-00/VisionCopilot/pharmacy/images"):
            frame = cv2.imread(r"/home/portable-00/VisionCopilot/pharmacy/images/"+filename)
            prescription, res_array = self.yoloc_decctor.scan_prescription(frame)
            print(prescription)
test = MainProcess()
test.run()