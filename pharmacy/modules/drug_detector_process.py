import multiprocessing
import numpy as np


class DrugDetectorProcess(multiprocessing.Process):
    def __init__(self, inference_event: multiprocessing.Event, done_barrier: multiprocessing.Barrier, frame_shared_array: multiprocessing.Array, drug_detection_outputs: multiprocessing.Queue) -> None:

        self.inference_event = inference_event
        self.done_barrier = done_barrier
        self.frame_shared_array = frame_shared_array
        self.drug_detection_outputs = drug_detection_outputs

        ############### YOUR CODE HERE ###############
        
        
        ############### YOUR CODE HERE ###############

        super().__init__()

    def run(self):
        while True:
            self.inference_event.wait()
            self.execute()
    
    def execute(self) -> None:
        image = np.frombuffer(self.frame_shared_array.get_obj(), dtype=np.uint8).reshape((1080, 1920, 3))
        
        ############### YOUR CODE HERE ###############
        
        
        ############### YOUR CODE HERE ###############
        
        self.drug_detection_outputs.put('drug')
        
        self.done_barrier.wait()
        self.inference_event.clear()