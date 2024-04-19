import numpy as np
from copy import deepcopy
from typing import List, Dict
from modules.object_tracker import ObjectTracker
from qinglang.dataset.utils.utils import xywh2xyxy, check_bboxes_intersection
from qinglang.utils.mathematic import euclidean_distance
from qinglang.utils.utils import Config


class CatchChecker:
    def __init__(self) -> None:
        self.config = Config('configs/catch_check.yaml')

        self.hand_tracker = ObjectTracker()
        self.medicine_tracker = ObjectTracker()

    def observe(self, hands: List[Dict], medicines: List[Dict]) -> None:
        self.hand_tracker.update(deepcopy(hands))
        self.medicine_tracker.update(deepcopy(medicines))
    
    def check(self) -> List:
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
