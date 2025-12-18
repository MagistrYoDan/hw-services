from fastapi import APIRouter
import onnx
from pathlib import Path

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "model.onnx"

@router.get("/metadata")
def get_metadata():
    model = onnx.load(MODEL_PATH)
    return {m.key: m.value for m in model.metadata_props}
