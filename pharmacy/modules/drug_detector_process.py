import numpy as np
import multiprocessing as multiprocessing
from modules.yoloc_dector import YolovDector
from qinglang.dataset.utils.utils import centerwh2xywh
from utils.yolv_infer.index_transfer import IndexTransfer
from qinglang.utils.utils import Config


class DrugDetectorProcess(multiprocessing.Process):
    def __init__(self, init_done_barrier: multiprocessing.Barrier, inference_event: multiprocessing.Event, done_barrier: multiprocessing.Barrier, terminal_event: multiprocessing.Event, frame_shared_array: multiprocessing.Array, drug_detection_outputs: multiprocessing.Queue) -> None:
        super().__init__()
        
        self.init_done_barrier = init_done_barrier
        self.inference_event = inference_event
        self.done_barrier = done_barrier
        self.terminal_event = terminal_event
        self.frame_shared_array = frame_shared_array
        self.drug_detection_outputs = drug_detection_outputs
        
        ############### YOUR CODE HERE ###############
        self.index_transfer = IndexTransfer()
        self.source = Config("configs/source.yaml")
        self.medcine_detect = YolovDector()
        self.save_folder_path = self.source.save_folder_path
        ############### YOUR CODE HERE ###############

        self.daemon = True

    def run(self):
        self.init_done_barrier.wait()
        
        while True:
            self.inference_event.wait()
            
            if self.terminal_event.is_set():
                break
            
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
 
        self.drug_detection_outputs.put(yolo_results_list)
        ############### YOUR CODE HERE ###############
        self.done_barrier.wait()
        self.inference_event.clear()