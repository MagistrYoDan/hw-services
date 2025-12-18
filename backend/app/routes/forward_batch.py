from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import json
from app.inference import predict_from_json

router = APIRouter()

class BatchRequest(BaseModel):
    features: List[List[float]]

@router.post("/forward_batch")
def forward_batch(req: BatchRequest):
    try:
        results = []
        for row in req.features:
            res = predict_from_json(json.dumps([row]))
            results.append(res)
        return {"results": results}
    except Exception as e:
        print("Batch inference error:", e)
        raise HTTPException(403, "модель не смогла обработать данные")
