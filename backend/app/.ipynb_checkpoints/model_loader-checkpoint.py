import onnxruntime as ort

def load_model():
    return ort.InferenceSession("backend/models/model.onnx")
