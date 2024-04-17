import multiprocessing
import numpy as np
from modules.yoloc_dector import YolovDector
from qinglang.dataset.utils.utils import plot_xywh,centerwh2xywh

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
                print(ocr_list)
        if yolo_detect_results is not None:
            clss = yolo_detect_results[0]
            bboxs = yolo_detect_results[1]
            for i in range(len(clss)):        
                self.drug_detection_outputs.put({'bbox': centerwh2xywh((bboxs[i])[:4]), 'category_id': clss[i]})
            
        #    self.drug_detection_outputs = detectresults 
        ############### YOUR CODE HERE ###############
        self.drug_detection_outputs.put('drug') # change to your detection result in format [{'bbox': centerwh2xywh(bbox[:4]), 'category_id': 0} for bbox in bboxes]
        self.done_barrier.wait()
        self.inference_event.clear()