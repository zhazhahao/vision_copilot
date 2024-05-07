# Benchmark

## Metrics

### Object Detection
* Accuracy：预测药品中所有预测正确的药品的比例。
* precision：预测为某category_id的物体里面真的是category_id类药品的比例。
* recall：所有标注过category_id中预测正确的比例。
### Action Recognition
#### Qualitative Evaluation 
* 检出率：一个动作正在发生，且被识别的百分比。
* 分均误报：一个动作没有发生，但被识别到了，平均每分钟发生的次数。

#### Quantitative Evaluation
* IoU@0.5