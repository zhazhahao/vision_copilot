import numpy as np
import torch.multiprocessing as multiprocessing
from modules.cameras import VirtualCamera
from modules.catch_checker import CatchChecker
from modules.drug_detector_process import DrugDetectorProcess
from modules.hand_detector_process import HandDetectorProcess
from modules.detect_results_stdout import stdrefresh
from qinglang.utils.utils import ClassDict, Config


class MainProcess:
    def __init__(self) -> None:
        self.source = Config("pharmacy/configs/source.yaml")
        multiprocessing.set_start_method('spawn')
        self.stream = VirtualCamera(self.source.virtual_camera_source)
        self.init_shared_variables()
        self.init_subprocess()
        self.catch_checker = CatchChecker()

        self.stop_flag = False

    
    def init_shared_variables(self):
        self.frame_shared_array = multiprocessing.Array('B', 1920 * 1080 * 3)
        self.inference_event = multiprocessing.Event()
        self.done_barrier = multiprocessing.Barrier(3)
        self.hand_detection_queue = multiprocessing.Queue()
        self.drug_detection_queue = multiprocessing.Queue()

    def init_subprocess(self) -> None:
        self.subprocesses = ClassDict(
            drug_detector = DrugDetectorProcess(self.inference_event, self.done_barrier, self.frame_shared_array, self.drug_detection_queue),
            hand_detector = HandDetectorProcess(self.inference_event, self.done_barrier, self.frame_shared_array, self.hand_detection_queue),
        )

        for p in self.subprocesses.values():
            p.start()

    def run(self):
        for frame in self.stream:

            if self.stop_flag:
                break
            
            self.share_frame(frame)
            
            hand_detection_results, drug_detection_results = self.parallel_inference()
            
            print(hand_detection_results)
            print(drug_detection_results)

            self.catch_checker.observe(hand_detection_results, drug_detection_results)
            check_results = self.catch_checker.check()

            print(check_results)
            self.update(frame,check_results)

    
    def update(self,frame,check_results):
        pass

    def share_frame(self, frame: np.ndarray) -> None:
        np.copyto(np.frombuffer(self.frame_shared_array.get_obj(), dtype=np.uint8), frame.flatten())

    def parallel_inference(self):
        # start parallel inference
        self.inference_event.set()
        self.done_barrier.wait()
        
        # finish parallel inference
        self.inference_event.clear()
        self.done_barrier.reset()
        
        # get inference results
        hand_detection_results = self.hand_detection_queue.get()
        drug_detection_results = self.drug_detection_queue.get()
        
        return hand_detection_results, drug_detection_results


if __name__ == '__main__':
    # multiprocessing.set_start_method('spawn')

    test1 = MainProcess()
    test1.run()
