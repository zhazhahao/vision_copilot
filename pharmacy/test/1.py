import onnx
import onnxoptimizer
onnx_model = onnx.load("test.onnx")  # load onnx model
model_simp, check = onnxoptimizer.optimize(onnx_model)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, "re.onnx")
print('finished exporting onnx')