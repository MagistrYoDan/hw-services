import onnxruntime as ort

def load_model():
    return ort.InferenceSession("models/model.onnx")
