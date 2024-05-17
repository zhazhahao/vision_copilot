from collections import Counter
import time
import cv2
import numpy as np
import torch.multiprocessing as multiprocessing
from utils.ocr_infer.prescription_utils import FrameMaxMatchingCollections, PreScriptionRecursiveObject


class OCRProcess(multiprocessing.Process):
    def __init__(self, init_done_barrier: multiprocessing.Barrier, inference_event: multiprocessing.Event, done_barrier: multiprocessing.Barrier, terminal_event: multiprocessing.Event, frame_shared_array: multiprocessing.Array, wild_ocr_outputs: multiprocessing.Queue) -> None:
        super().__init__()
        self.init_done_barrier = init_done_barrier
        self.inference_event = inference_event
        self.done_barrier = done_barrier
        self.terminal_event = terminal_event
        self.frame_shared_array = frame_shared_array
        self.wild_ocr_outputs = wild_ocr_outputs        
        self.daemon = True
        
    def init_process(self):
        from modules.ocr_detector import OcrDector
        self.ocr_detector = OcrDector()
        self.finish_candidate = False

        
    def scan_prescription(self,stream):
        end_trigger_times = 0
        times = 0
        loss_track_threshold = 0
        status_dict = {}
        res_counter = []
        counter = Counter()
        for frame in stream:
            times += 1
            if not self.finish_candidate:
                (dt_box_res,prescription,trigger) = self.ocr_detector.procession(frame,"prescription")
                # print(prescription)
                if "领退药药单汇总" in prescription or "统领单(针剂)汇总" in prescription:
                    self.candiancate.update(prescription,times)
                    self.finish_candidate = True
                continue
            (dt_box_res,prescription,trigger) = self.ocr_detector.procession(frame,"prescription")
            end_trigger_times += 1 if trigger else 0
            counter.update(prescription)
            res_frame , res_counter= frame , [dt_box_res,prescription]
            height,width = self.ocr_detector.getavgSize(res_counter[0])
            for res in res_counter[0]:
                res_frame = cv2.rectangle(res_frame, tuple(res[0].astype("int")),tuple(res[2].astype("int")),color=(0, 255, 0),thickness=-1)
            conter_len = 0 
            min_width = 1920
            for i in range(1, len(res_counter[0])):
                if (res_counter[0][i][3][1] - res_counter[0][i-1][0][1]) >= height * 1.5:
                    fix_height = res_counter[0][i-1][0][1]
                    first_set = True
                    while fix_height < res_counter[0][i][0][1]:
                        selected_height = min(int(res_counter[0][i - 1][2][1]),int(res_counter[0][i - 1][3][1])) if first_set else int(selected_height + height)
                        selected_width  = min(int(res_counter[0][i - 1][3][0]),int(res_counter[0][i][3][0]))
                        min_width = min(min_width,selected_width)
                        first_set = False
                        if selected_height + int(height * 1.5) > res_frame.shape[0]:
                            selected_height = res_frame.shape[0] - int(height * 1.5)
                        rec_res = self.ocr_detector.procession(res_frame[selected_height + int(height * 0.5):selected_height + int(height * 1.5),
                                                   selected_width:selected_width + int(width * 1.5)],options="Single")
                        # print(prescription)
                        status_dict[rec_res] = "Update"
                        fix_height += height
                        if rec_res == None:
                            fix_height += height
                            continue
                        conter_len += 1
                        counter.update([rec_res])
                        res_counter[1].insert(i - 1 + conter_len,rec_res)
            if res_counter[0].__len__() != 0:
                selected_height = min(int(res_counter[0][0][0][1]),int(res_counter[0][0][1][1])) 
                selected_width  = int(res_counter[0][0][2][0])                         
                if selected_height - int(height) * 1.5 < 0:
                    continue
                if selected_height - int(height * 1) < 0:
                    selected_height = int(height)
                if selected_width - int(width) < 0:
                    selected_width = int(width)
                if selected_width + int(width) > 1920:
                    selected_width = 1920 - int(width)

                rec_res = self.ocr_detector.procession(res_frame[selected_height - int(height * 1):selected_height ,
                                           selected_width - int(width):selected_width + int(width)]
                                 ,options="Single")
                if rec_res != None:
                    res_counter[1].insert(0,rec_res)
            self.frame_collections.update(frame,max_candicated=[dt_box_res,prescription],times=times)
            # print(end_trigger_times)  
            # print(loss_track_threshold)
            if res_counter[1].__len__() > 0:
                self.candiancate.update(res_counter[1],times)
                loss_track_threshold = 0
            else:
                loss_track_threshold += 1
            if loss_track_threshold > self.loss_track_threshold:
                break
            if end_trigger_times == self.max_opportunity:
                # print(times)
                break
        # print(prescription)
        tools = self.frame_collections.values()
        for key,values in tools.items():
            self.candiancate._check_merge_drugs([key,values["max_candicated"][1]])
        try:
            max_count = np.median([value for value in counter.values() if value != "领退药药单汇总"])
            max_counts = np.median([answer.counts for answer in self.frame_collections.result_counter.values()])
        except:
            return None
        recursive_con = [answer 
            for answer in self.candiancate.recursive_obj
           if counter[answer] >= max_count / 4 or (answer in status_dict and self.frame_collections.result_counter[answer].counts >= max_counts / 4)]
     
        final_con = [answer 
                    for obj in self.candiancate.static_obj  
                    for answer in obj[1]
                   if counter[answer] >= max_count / 4 or (answer in status_dict and self.frame_collections.result_counter[answer].counts >= max_counts / 4)]
        
        recursive_con.extend(final_con)
        answer_dict = self.ocr_detector.group_similar_strings(list(set(recursive_con)),self.frame_collections.result_counter)
        # print(answer_dict)
        # print(ans for answer_res in answer_dict
        #         for ans in answer_res)
        return [ans for answer_res in answer_dict
                for ans in answer_res if ans != "统领单(针剂)汇总" or ans != "领退药药单汇总"]
        
    def run(self) -> None:
        self.init_process()
        
        self.init_done_barrier.wait()
        
        while True:
            self.inference_event.wait()
            
            if self.terminal_event.is_set():
                break
            
            self.execute()

    
    def execute(self) -> None:
        image = np.frombuffer(self.frame_shared_array.get_obj(), dtype=np.uint8).reshape((1080, 1920, 3))
        result = self.ocr_detector.ocr_detect(image)
        self.wild_ocr_outputs.put(result)
        self.done_barrier.wait()
        self.inference_event.clear()