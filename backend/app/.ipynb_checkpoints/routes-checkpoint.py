from fastapi import APIRouter, Header, HTTPException
from app.inference import predict_from_json

router = APIRouter()

@router.post("/forward")
def forward(x_features: str | None = Header(None)):
    if not x_features:
        raise HTTPException(400, "bad request")

    try:
        return predict_from_json(x_features)
    except Exception:
        raise HTTPException(403, "модель не смогла обработать данные")
