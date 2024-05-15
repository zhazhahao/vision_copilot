from qinglang.utils.utils import dump_yaml

config = {
    'datasets': [
        {
            'root_path': "/home/portable-00/data/datasets/20240313_160556",
            'video': '20240313_160556.mp4',
            'prescription': 'prescription.json',
            'catch_annotation': 'catch_annotation.json',
        },
    ]
}

dump_yaml(config, 'configs/benchmark.yaml')