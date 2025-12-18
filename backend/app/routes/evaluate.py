from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
import json
from sklearn.metrics import accuracy_score, f1_score
from app.inference import predict_from_json

router = APIRouter()

"""
@router.post("/evaluate")
async def evaluate(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV supported")

    df = pd.read_csv(file.file)

    if "target" not in df.columns:
        raise HTTPException(400, "CSV must contain target column")

    y_true = df["target"].tolist()
    X = df.drop(columns=["target"]).values.tolist()

    preds = []
    for row in X:
        res = predict_from_json(json.dumps([row]))
        preds.append(res["label"][0])

    return {
        "metrics": {
            "accuracy": accuracy_score(y_true, preds),
            "f1": f1_score(y_true, preds, average="weighted")
        },
        "predictions": preds
    }
"""

@router.post("/evaluate")
async def evaluate(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV supported")

    df = pd.read_csv(file.file)

    if "target" not in df.columns:
        raise HTTPException(400, "CSV must contain target column")

    y_true = df["target"].astype(int).tolist()
    X = df.drop(columns=["target"]).astype(float).values.tolist()

    # Один вызов модели
    result = predict_from_json(json.dumps(X))
    preds = result["label"]

    return {
        "metrics": {
            "accuracy": accuracy_score(y_true, preds),
            "f1": f1_score(y_true, preds, average="weighted")
        },
        "predictions": preds
    }

