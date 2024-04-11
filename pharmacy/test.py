from qinglang.utils.utils import Config

config = Config(
    tracking_depth = 120,
    decay_time = 5,
    translation_threshold = 100,
)

config.dump('object_tracking.yaml')
