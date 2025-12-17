import numpy as np
import json
from app.model_loader import load_model

session = load_model()

def predict_from_json(payload: str):
    data = np.array(json.loads(payload), dtype=np.float32)
    if data.ndim != 2:
        raise ValueError("Invalid input shape")

    preds = session.run(None, {"input": data})[0]
    return {
        "prediction": preds.argmax(axis=1).tolist(),
        "raw_output": preds.tolist()
    }
