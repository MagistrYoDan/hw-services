from fastapi import APIRouter
import onnx

router = APIRouter()

@router.get("/metadata")
def metadata():
    model = onnx.load("backend/models/model.onnx")
    return {m.key: m.value for m in model.metadata_props}
