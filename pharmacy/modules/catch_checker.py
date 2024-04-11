import numpy as np
from typing import List
from qinglang.object_detection.object_tracker import ObjectTracker
from qinglang.dataset.utils.utils import xywh2xyxy, check_bboxes_intersection
from qinglang.utils.utils import Config
from qinglang.utils.mathematic import euclidean_distance


class CatchChecker:
    def __init__(self) -> None:
        self.hand_tracker = ObjectTracker(120, 5, 200)
        self.medicine_tracker = ObjectTracker(120, 5, 200)
    
        self.config = Config('/home/portable-00/VisionCopilot/pharmacy/configs/catch_check.yaml')

    def observe(self, hands: List, medicines: List) -> None:
        self.hand_tracker.update(hands)
        self.medicine_tracker.update(medicines)
    
    def check(self) -> bool:
        medicines_catched = []
        for hand in self.hand_tracker.tracked_objects:
            hand_bboxes = [node and xywh2xyxy(node['bbox']) for node in hand.trajectory]

            candidates = []
            for medicine in self.medicine_tracker.tracked_objects:
                medicine_bboxes = [node and xywh2xyxy(node['bbox']) for node in medicine.trajectory]
                
                # Check bboxes must intersection in latest {missing_tolerance} frames
                if not any([check_bboxes_intersection(hand_bbox, medicine_bbox) if hand_bbox is not None and medicine_bbox is not None else False for hand_bbox, medicine_bbox in zip(hand_bboxes[:self.config.missing_tolerance], medicine_bboxes[:self.config.missing_tolerance])]):
                    continue
                
                # Check if number of intersection frames in {search_depth} frames larger than {catch_threshold}
                if [check_bboxes_intersection(hand_bbox, medicine_bbox) if hand_bbox is not None and medicine_bbox is not None else False for hand_bbox, medicine_bbox in zip(hand_bboxes[:self.config.search_depth], medicine_bboxes[:self.config.search_depth])].count(True) >= self.config.catch_threshold:
                    candidates.append(medicine)
            
            if candidates:
                medicines_catched.append(min(candidates, key=lambda medicine: np.mean([euclidean_distance(hand_node.speed, medicine_node.speed) for hand_node, medicine_node in zip(hand.trajectory[:self.config.speed_search_depth], medicine.trajectory[:self.config.speed_search_depth]) if hand_node and medicine_node])))
        
        return medicines_catched


if __name__ == '__main__':
    ...
