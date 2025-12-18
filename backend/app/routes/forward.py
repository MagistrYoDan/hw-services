from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import json
from app.inference import predict_from_json

router = APIRouter()

class ForwardRequest(BaseModel):
    features: List[float]

@router.post("/forward")
def forward(req: ForwardRequest):
    try:
        return predict_from_json(json.dumps([req.features]))
    except Exception as e:
        print("Inference error:", e)
        raise HTTPException(403, "модель не смогла обработать данные")
