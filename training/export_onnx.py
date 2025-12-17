import pickle
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx
from datetime import datetime
import subprocess

with open("artifacts/model.pkl", "rb") as f:
    model = pickle.load(f)

initial_type = [("input", FloatTensorType([None, 10]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

def add_meta(key, value):
    meta = onnx_model.metadata_props.add()
    meta.key = key
    meta.value = value

add_meta("commit_hash", subprocess.getoutput("git rev-parse HEAD"))
add_meta("created_at", datetime.utcnow().isoformat())
add_meta("experiment_name", "rf_tabular_baseline")

onnx.save(onnx_model, "artifacts/model.onnx")

print("✅ ONNX модель сохранена с метаданными")
