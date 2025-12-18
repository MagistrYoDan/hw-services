from pathlib import Path
import onnxruntime as ort

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "model.onnx"

def load_model():
    return ort.InferenceSession(str(MODEL_PATH))

