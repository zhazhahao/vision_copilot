import numpy as np
import torch.multiprocessing as multiprocessing
from modules.yoloc_dector import YolovDector
from qinglang.dataset.utils.utils import centerwh2xywh


class DrugDetectorProcess(multiprocessing.Process):
    def __init__(self, inference_event: multiprocessing.Event, done_barrier: multiprocessing.Barrier, frame_shared_array: multiprocessing.Array, drug_detection_outputs: multiprocessing.Queue) -> None:

        self.inference_event = inference_event
        self.done_barrier = done_barrier
        self.frame_shared_array = frame_shared_array
        self.drug_detection_outputs = drug_detection_outputs

        ############### YOUR CODE HERE ###############
        self.medcinedetect = YolovDector()
        
        ############### YOUR CODE HERE ###############

        super().__init__()

    def run(self):
        while True:
            self.inference_event.wait()
            self.execute()
    
    
    def execute(self) -> None:
        image = np.frombuffer(self.frame_shared_array.get_obj(), dtype=np.uint8).reshape((1080, 1920, 3))
        
        ############### YOUR CODE HERE ###############
        yolo_detect_results = self.medcinedetect.yolo_detect(frame=image) # return [list(cls), list(xywh)]
        ocr_detect_results = self.medcinedetect.ocr_detect(frame=image) # return [{med_json} or none]
        ocr_list = []
        for i in range(len(ocr_detect_results)):
            if ocr_detect_results[i] is not None:
                ocr_list.append(ocr_detect_results[i].get("药品名称"))
           
        yolo_results_list = [] 
        
        if yolo_detect_results is not None:
            
            clss = yolo_detect_results[0]
            bboxes = yolo_detect_results[1]
            
            yolo_results_list = [{'bbox': centerwh2xywh(bboxes[i]), 'category_id': int(clss[i])}  for i in range(len(bboxes))]
                
        self.drug_detection_outputs.put(yolo_results_list)
        ############### YOUR CODE HERE ###############

        self.done_barrier.wait()
        self.inference_event.clear()