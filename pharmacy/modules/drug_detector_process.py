import os
import time
import cv2
import numpy as np
import torch.multiprocessing as multiprocessing
from modules.yoloc_dector import YolovDector
from qinglang.dataset.utils.utils import centerwh2xywh
from utils.yolv_infer.index_transfer import IndexTransfer
from qinglang.utils.utils import Config

class DrugDetectorProcess(multiprocessing.Process):
    def __init__(self, inference_event: multiprocessing.Event, done_barrier: multiprocessing.Barrier, frame_shared_array: multiprocessing.Array, drug_detection_outputs: multiprocessing.Queue) -> None:

        self.inference_event = inference_event
        self.done_barrier = done_barrier
        self.frame_shared_array = frame_shared_array
        self.drug_detection_outputs = drug_detection_outputs
        self.index_transfer = IndexTransfer()
        ############### YOUR CODE HERE ###############
        self.source = Config("configs/source.yaml")
        self.medcine_detect = YolovDector()
        self.save_folder_path = self.source.save_folder_path
        ############### YOUR CODE HERE ###############

        super().__init__()

    def run(self):
        while True:
            self.inference_event.wait()
            self.execute()
    
    
    def execute(self) -> None:
        image = np.frombuffer(self.frame_shared_array.get_obj(), dtype=np.uint8).reshape((1080, 1920, 3))
        
        ############### YOUR CODE HERE ###############
        yolo_detect_results = self.medcine_detect.yolo_detect(frame=image) # return [list(cls), list(xywh)]  
        yolo_results_list = [] 
        
        if yolo_detect_results is not None:
            
            clss = yolo_detect_results[0]
            bboxes = yolo_detect_results[1]
            
            yolo_results_list = [{'bbox': centerwh2xywh(bboxes[i]), 'category_id': int(clss[i])}  for i in range(len(bboxes))]
            
            for i in range(len(bboxes)):
                timestamp = int(time.time())
                unique_filename = f'image_with_bbox_{timestamp}.jpg'
                save_path = os.path.join(self.save_folder_path, unique_filename)
                self.medcine_detect.plot_save(image, centerwh2xywh(np.array(bboxes[i], dtype=int)), color=(0, 0, 255), save_path=save_path, text = self.index_transfer.cls2name(index = int(clss[i])))  
        self.drug_detection_outputs.put(yolo_results_list)
        ############### YOUR CODE HERE ###############
        self.done_barrier.wait()
        self.inference_event.clear()