import os
import numpy as np
from modules.camera_processor import CameraProcessor
from qinglang.data_structure.video.video_base import VideoFlow
from qinglang.utils.utils import ClassDict


class CameraBase:
    def __init__(self) -> None:
        ...

    def __iter__(self) -> object:       
        ...

    def __next__(self) -> np.ndarray:
        ...
    
    def beep(self) -> None:
        ...

    def release(self) -> None:
        ...


class VirtualCamera(VideoFlow, CameraBase):
    def __init__(self, source: str) -> None:
        super().__init__(source)
    
    def __iter__(self) -> object:
        return super().__iter__()
    
    def __next__(self) -> np.ndarray:
        return super().__next__()


class DRIFTX3(CameraBase):
    def __init__(self) -> None:
        self.videoCapture = None

    def __iter__(self):
        self.videoCapture = CameraProcessor()
        return self

    def __next__(self) -> np.ndarray:
        ret, frame = self.videoCapture.achieve_image()
        while not ret:
            ret, frame = self.videoCapture.achieve_image()
        return frame

    def beep(self):
        self.videoCapture.send_wrong()

    def release(self) -> None:
        return self.videoCapture.end_process()


if __name__ == '__main__':
    vc = VirtualCamera("/mnt/nas/datasets/Pharmacy_for_label/20240313/20240313_160556/20240313_160556.mp4")
    for frame in vc:
        print(frame)
    vc.beep()
