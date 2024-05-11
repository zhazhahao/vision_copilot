import json
import cv2
import numpy as np
from itertools import groupby
from typing import Dict, List
from ultralytics import YOLO
from engine import Engine
from modules.cameras import VirtualCamera
from qinglang.utils.utils import Config, ClassDict, load_json, load_yaml


class BenchmarkEngine(Engine):
    def __init__(self, logger: ClassDict) -> None:
        super().__init__()
        
        self.logger = logger

    def post_process(self, frame, check_results, hand_detection_results, drug_detection_results, hand_tracked, drug_tracked):
        self.logger.check_results.append(check_results)
        self.logger.hand_detection_results.append(hand_detection_results)
        self.logger.drug_detection_results.append(drug_detection_results)


class PharmacyCopilotBenchmark:
    def __init__(self) -> None:
        self.config = Config("configs/benchmark.yaml")
        
    def benchmark(self, dataset: ClassDict) -> None:
        logger = ClassDict(
            check_results = [],
            hand_detection_results = [],
            drug_detection_results = [],
        )
        
        engine = BenchmarkEngine(logger)
        engine.stream = VirtualCamera(dataset.video)
        engine.prescription = load_yaml(dataset.prescription)
        
        annotations = load_json(dataset.annotations)
        catch_checking_sequence: List = ...
        
        engine.run()
        
        logger = ClassDict(**{key: np.array(value) for key, value in logger})

        # catch checking examination
        catch_count = 0
        catch_tp_count = 0
        catch_fp_count = 0
        
        for drug_id, frame_idxes in self.group_consecutive_elements(catch_checking_sequence):
            predictions = {element: count for element, count in zip(np.unique(logger.check_results[frame_idxes], return_counts=True))}

            if drug_id != None:
                catch_count += 1
                
                if predictions.get(drug_id):
                    catch_tp_count += 1
            
            catch_fp_count += len([key for key in predictions.keys() if key not in [None, drug_id]])
        
        catch_checking_recall = catch_tp_count / catch_count
            
    def group_consecutive_elements(lst):
        groups = []
        current_index = 0
        for key, group in groupby(lst):
            group_length = len(list(group))
            groups.append((key, current_index + np.arange(group_length)))
            current_index += group_length
        return groups


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
        annotations_dict,annotations_lenth = self.load_annotations()
        Tp = 0
        Tn = 0
        Postive = 0
        Negative = 0
        Fn = 0
        Fp = 0
        correct_count = 0
        error_frames = []
        
        for frame_idx in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            # 假设模型推断函数是 predict_frame，返回预测结果
            results= self.model(frame, verbose=False)
            for result in results:
                boxes = result.boxes
                cls = boxes.cls
                if cls.numel() == 0:
                    if annotations_dict[frame_idx] == -1:
                        Tn += 1
                        Negative += 1
                        continue
                    else:
                        Fn += 1
                        Postive += 1
                        error_frames.append(frame_idx)
                else:
                    if annotations_dict[frame_idx] == -1:
                        Negative += 1
                        Fp += 1
                        continue
                    if int(cls[0].item()) == annotations_dict[frame_idx]:
                        Postive += 1
                        Tp += 1
                    else:
                        Fp += 1
                        Postive += 1
                        error_frames.append(frame_idx)

        accuracy = (Tp + Tn) / (Tp + Tn + Fp + Fn)
        precision = Tp / (Tp + Fp)
        recall = Tp / Postive
        error_rate = 1 - accuracy
        print(annotations_lenth)
        print(correct_count)
        print(len(error_frames))
        print("precision:" + str(precision))
        print("recall:" + str(recall))

        return accuracy, error_rate, error_frames
    
    # list[clsids]
    def load_annotations(self):
        with open(self.json_path, 'r') as file:
            data = json.load(file)  # 正确地加载JSON文件
        annotations_dict = {}
        annotations_lenth = 0
        # 遍历images列表中的每个图像对象
        for annotations in data["annotations"]:
            # 使用图像的id作为键，标注的id列表作为值
            annotations_dict[int(annotations["image_id"])] = int(annotations["category_id"])
            annotations_lenth += 1
        # 获取视频的总帧数
        cap, _, _ = self.load_video()
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()  

        # 初始化一个与视频帧数相同长度的列表，并填充默认值
        image_annotation_list = [-1] * frame_count

        # 使用图像ID作为索引赋值
        for image_id, annotation_id in annotations_dict.items():
            if image_id < frame_count:  # 确保ID在范围内
                image_annotation_list[image_id] = annotation_id

        return image_annotation_list,annotations_lenth
                


# 示例用法
test = Benchmark()
accuracy, error_rate, error_frames =  test.benchmark_model()
print("Accuracy:", accuracy)
# print("Error rate:", error_rate)
# print("Error frames:", error_frames)
