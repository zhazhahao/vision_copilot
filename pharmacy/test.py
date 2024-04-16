from qinglang.dataset.utils.utils import plot_xywh, centerwh2xywh

bbox = [100, 200, 20, 30]
print(centerwh2xywh(bbox))
