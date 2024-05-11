import numpy as np
from qinglang.utils.utils import dump_json
from qinglang.data_structure.video.video_base import VideoFlow


video_src = "/home/portable-00/data/20240313_160556/20240313_160556.mp4"
catch_actions = [
    {
        "category_id": 84,
        "time_span": [50, 125],
    },
    {
        "category_id": 82,
        "time_span": [170, 240],
    },
    {
        "category_id": 89,
        "time_span": [285, 408],
    },
    {
        "category_id": 92,
        "time_span": [470, 560],
    },
    {
        "category_id": 107,
        "time_span": [655, 793],
    },
    {
        "category_id": 110,
        "time_span": [820, 938],
    },
    {
        "category_id": 111,
        "time_span": [956, 1015],
    },
    {
        "category_id": 106,
        "time_span": [1190, 1308],
    },
    {
        "category_id": 94,
        "time_span": [1347, 1448],
    },
    {
        "category_id": 93,
        "time_span": [1497, 1580],
    },
]

video_flow = VideoFlow("/home/portable-00/data/20240313_160556/20240313_160556.mp4")

start_frame = 0
end_frame = video_flow.metainfo.total_frames

switch_points = [point for action in catch_actions for point in action['time_span']]

frame_annotations = np.zeros(end_frame, dtype=int)

for action in catch_actions:
    frame_annotations[range(*action['time_span'])] = action['category_id']

# dump_json(frame_annotations.tolist(), "/home/portable-00/data/20240313_160556/catch_action.json")
print(end_frame)