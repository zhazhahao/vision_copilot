import os
import json
import cv2
import time
import pickle
import numpy as np
import matplotlib
from datetime import datetime
from itertools import groupby
from typing import Dict, List
from ultralytics import YOLO
from engine import Engine
from tqdm import tqdm
from modules.cameras import VirtualCamera
from qinglang.utils.utils import Config, ClassDict, Logger, load_json, load_yaml
from qinglang.utils.io import load_pickle, dump_pickle, clear_lines

matplotlib.use('agg')

"""
TO BE DONE:
  - multi-dataset benchmark support (√)
  - save noticeable frames automatically

"""


class BenchmarkEngine(Engine):
    def __init__(self) -> None:
        super().__init__()
        
    def post_process(self, frame, check_results, hand_detection_results, drug_detection_results, hand_tracked, drug_tracked):
        self.history.check_results.append(check_results and [{'category_id': check_results[0].category_id, 'bbox': check_results[0].get_latest_valid_node().bbox}])
        self.history.hand_detection_results.append(hand_detection_results)
        self.history.drug_detection_results.append(drug_detection_results)

        self.pbar.update(1)


class PharmacyCopilotBenchmark:
    def __init__(self) -> None:
        self.config = Config("configs/benchmark.yaml")
        self.logger = Logger()
    
        self._init_work_dir()

    def _init_work_dir(self) -> None:
        self.work_dir = f"benchmark_results/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        os.makedirs(self.work_dir, exist_ok=True)

    def run(self) -> None:
        for dataset in self.config.datasets:
            self.benchmark(ClassDict(dataset))

    def benchmark(self, dataset: ClassDict) -> None:
        self.logger.info(rf"Start benchmark video {os.path.join(dataset.root_path, dataset.video)}.")
        
        # Init work directory
        dataset.work_dir = os.path.join(self.work_dir, os.path.basename(dataset.root_path))
        os.makedirs(dataset.work_dir, exist_ok=True)
        
        # Inference
        self.logger.info(rf"Inference started.")

        results = self.inference(dataset)
        dump_pickle(results, os.path.join(dataset.root_path, 'results.pkl'))
        
        # results = load_pickle(os.path.join(dataset.root_path, 'results.pkl'))
        
        # Benchmark        
        self.test_catch_recognition(dataset, results)

    def inference(self, dataset: ClassDict) -> ClassDict:
        history = ClassDict(
            check_results = [],
            hand_detection_results = [],
            drug_detection_results = [],
        )

        engine = BenchmarkEngine()
        engine.stream = VirtualCamera(os.path.join(dataset.root_path, dataset.video))
        engine.pbar = tqdm(total=len(engine.stream), desc="Inference progress:",colour="green")
        engine.history = history
        
        engine.run()
        
        return history

    def test_catch_recognition(self, dataset: ClassDict, results: ClassDict):
        # Init
        prescription = load_yaml(os.path.join(dataset.root_path, dataset.prescription))
        drugs_caught = set()
        
        catch_annotations = np.array(load_json(os.path.join(dataset.root_path, dataset.catch_annotation)))
        catch_predictions = np.array([0 if frame_result == [] else frame_result[0]['category_id'] for frame_result in results.check_results])

        catch_tp_count = 0
        catch_fp_count = 0
        
        # Check frames grouped by annotations
        for drug_id, [start, end] in self.group_consecutive_elements(catch_annotations):
            if drug_id == 0:
                continue
            
            if drug_id in catch_predictions[start: end]:
                catch_tp_count += 1
                
            # predictions = {element: count for element, count in zip(*np.unique([frame_result[0]['category_id'] for frame_result in results.check_results[start: end] if frame_result], return_counts=True))}
            # catch_fp_count += len([key for key in predictions.keys() if key not in [0, drug_id]])

        # Check frames grouped by predictions
        for drug_id, [start, end] in self.group_consecutive_elements(catch_predictions):
            if drug_id == 0:
                continue

            drugs_caught.add(drug_id)
            
            if not np.all((catch_annotations[start: end] == drug_id) | (catch_annotations[start: end] == 0)) or not np.any(catch_annotations[start: end] == drug_id):
                catch_fp_count += 1

        # calculate metrics
        catch_checking_recall = catch_tp_count / len(prescription)
        catch_fp_per_minute = catch_fp_count / len(catch_annotations) * 30
        drugs_missing = [drug for drug in prescription if drug not in drugs_caught]
        drugs_misreport = [drug for drug in drugs_caught if drug not in prescription]
        
        self.logger.info(rf"catch_checking_recall: {catch_checking_recall}, catch_fp_count: {catch_fp_count}, catch_fp_per_minute: {catch_fp_per_minute}, drugs_missing: {drugs_missing}, drugs_misreport: {drugs_misreport}")

    def group_consecutive_elements(self, lst):
        groups = []
        current_index = 0
        for key, group in groupby(lst):
            group_length = len(list(group))
            groups.append((key, [current_index, current_index + group_length]))
            # groups.append((key, current_index + np.arange(group_length)))
            current_index += group_length
        return groups


if __name__ == '__main__':
    benchmark = PharmacyCopilotBenchmark()
    benchmark.run()
    
