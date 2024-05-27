import numpy as np
from qinglang.utils.utils import dump_json
from qinglang.data_structure.video.video_base import VideoFlow

catch_actions = [
    {
        "category_id": 160,
        "time_span": [555, 600],
    },
    {
        "category_id": 160,
        "time_span": [689, 763],
    },
    {
        "category_id": 104,
        "time_span": [1091, 1100],
    },
    {
        "category_id": 104,
        "time_span": [1182, 1277],
    },
    {
        "category_id": 120,
        "time_span": [1536, 1620],
    },
    {
        "category_id": 316,
        "time_span": [2380, 2474],
    },
    {
        "category_id": 158,
        "time_span": [3130, 3232],
    },
    {
        "category_id": 158,
        "time_span": [3253, 3326],
    },
    {
        "category_id": 88,
        "time_span": [3615, 3689],
    },
    {
        "category_id": 88,
        "time_span": [3715, 3800],
    },
    {
        "category_id": 198,
        "time_span": [3940, 3978],
    },
    {
        "category_id": 197,
        "time_span": [4093, 4141],
    },
    {
        "category_id": 195,
        "time_span": [4215, 4303],
    },
    {
        "category_id": 173,
        "time_span": [4449, 4545],
    },
    {
        "category_id": 137,
        "time_span": [4951, 5060],
    },
    {
        "category_id": 140,
        "time_span": [5110, 5200],
    },
    {
        "category_id": 136,
        "time_span": [5391, 5460],
    },
    {
        "category_id": 101,
        "time_span": [5630, 5680],
    },
    {
        "category_id": 189,
        "time_span": [6155, 6244],
    },
]

video_flow = VideoFlow("/mnt/data/public/pharmacy_copilot/20240509/benchmark/0523_164718/20240523_164718.MP4")

start_frame = 0
end_frame = video_flow.metainfo.total_frames

switch_points = [point for action in catch_actions for point in action['time_span']]

frame_annotations = np.zeros(end_frame, dtype=int)

for action in catch_actions:
    frame_annotations[range(*action['time_span'])] = action['category_id']

dump_json(frame_annotations.tolist(), "/mnt/data/public/pharmacy_copilot/20240509/benchmark/0523_164718/catch_annotation.json")
dump_json(list(set([action["category_id"] for action in catch_actions])), "/mnt/data/public/pharmacy_copilot/20240509/benchmark/0523_164718/prescription.json")
# print(end_frame)