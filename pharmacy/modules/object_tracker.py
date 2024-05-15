import numpy as np
from typing import Union, List, Dict
from itertools import takewhile
from qinglang.data_structure.base import RollingArray
from qinglang.dataset.utils.utils import xywh2center
from qinglang.utils.mathematic import euclidean_distance
from qinglang.utils.utils import Config, ClassDict


class TrackedObject:
    def __init__(self, category_id: int, memory_depth: int) -> None:
        self.category_id = category_id
        self.trajectory = RollingArray(memory_depth)
        
    def add_frame_data(self, bbox: Union[None, List, np.ndarray]) -> None:
        data = None if bbox is None else ClassDict(
            bbox = bbox,
            center = xywh2center(bbox),
            speed = [0, 0] if len(self.trajectory) == 0 else (xywh2center(bbox) - self.get_latest_valid_node().center) / (1 + self.lost_tracking_counts()),
        )
        
        self.trajectory.put(data)

    def get_latest_valid_node(self) -> List:
        return next(node for node in self.trajectory if node is not None)
    
    def lost_tracking_counts(self) -> int:
        return len(list(takewhile(lambda x: x is None, self.trajectory)))


class ObjectTracker:
    def __init__(self) -> None:
        self.config = Config("configs/object_track.yaml")
        
        self.tracked_objects: List[TrackedObject] = []
        
    def update(self, detection_results: List[Dict]) -> None:
        self.tracked_objects = [tracked_object for tracked_object in self.tracked_objects if tracked_object.lost_tracking_counts() <= self.config.decay_time]

        for tracked_object in self.tracked_objects:
            target_bbox = tracked_object.get_latest_valid_node().bbox
            
            candidates = [detection_result for detection_result in detection_results if detection_result['category_id'] == tracked_object.category_id]
            candidates_distance = [euclidean_distance(xywh2center(target_bbox), xywh2center(candidate['bbox'])) for candidate in candidates]

            if candidates_distance == [] or (lambda nearest_idx: candidates_distance[nearest_idx])(nearest_idx := np.argmin(candidates_distance)) >= self.config.pixel_shift_threshold * (tracked_object.lost_tracking_counts() + 1):
                bbox = None
            else:
                bbox = candidates[nearest_idx]['bbox']
                detection_results.remove(candidates[nearest_idx])

            tracked_object.add_frame_data(bbox)
            
        for detection_result in detection_results:
            tracked_object = TrackedObject(detection_result['category_id'], self.config.tracking_depth)
            tracked_object.add_frame_data(detection_result['bbox'])
            
            self.tracked_objects.append(tracked_object)
            

if __name__ == '__main__':
    ...