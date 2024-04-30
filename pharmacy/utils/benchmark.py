import json
import cv2
from ultralytics import YOLO
from qinglang.utils.utils import Config


class Benchmark:
    def __init__(self):
        self.json_path = "/home/portable-00/VisionCopilot/pharmacy/database/downreal.json"
        self.source = Config("/home/portable-00/VisionCopilot/pharmacy/configs/source.yaml")
        self.model = YOLO(self.source.yolov_path)
        self.videopath = "/home/portable-00/data/video_0/20240313_160556.mp4"

    def load_video(self):
        # 加载视频，并返回视频的帧数和帧率
        cap = cv2.VideoCapture(self.videopath)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        return cap, frame_count, fps

    def benchmark_model(self):
        # 基准测试模型
        cap, frame_count, fps = self.load_video()
        annotations = self.load_annotations()
        
        correct_count = 0
        error_frames = []

        for frame_idx in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            # 假设模型推断函数是 predict_frame，返回预测结果
            results= self.model(frame)
            for result in results:
                boxes = result.boxes
                cls = boxes.cls
                if cls.numel() == 0:
                    continue
                else:
                    if int(cls[0].item()) == annotations[frame_idx]:
                        
                        correct_count += 1
                    else:
                        error_frames.append(frame_idx)

        accuracy = correct_count / frame_count
        error_rate = 1 - accuracy
        print(frame_count)
        print(correct_count)
        return accuracy, error_rate, error_frames
    
    # list[clsids]
    def load_annotations(self):
        with open(self.json_path, 'r') as file:
            data = json.load(file)  # 正确地加载JSON文件
        annotations = {}
        # 遍历images列表中的每个图像对象
        for image in data["images"]:
            # 使用图像的id作为键，标注的id列表作为值
            annotations[int(image["id"])] = int(image["annotation_ids"][0])

        # 获取视频的总帧数
        cap, _, _ = self.load_video()
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()  

        # 初始化一个与视频帧数相同长度的列表，并填充默认值
        image_annotation_list = [-1] * frame_count

        # 使用图像ID作为索引赋值
        for image_id, annotation_id in annotations.items():
            if image_id < frame_count:  # 确保ID在范围内
                image_annotation_list[image_id] = annotation_id

        return image_annotation_list
                


# 示例用法
test = Benchmark()
accuracy, error_rate, error_frames =  test.benchmark_model()
print("Accuracy:", accuracy)
print("Error rate:", error_rate)
print("Error frames:", error_frames)
