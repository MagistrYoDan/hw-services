import json
import numpy as np
from app.model_loader import load_model

session = load_model()

INPUT_NAME = session.get_inputs()[0].name
OUTPUT_NAMES = [o.name for o in session.get_outputs()]

def predict_from_json(payload: str):
    data = np.array(json.loads(payload), dtype=np.float32)

    if data.ndim != 2 or data.shape[1] != 10:
        raise ValueError("Invalid input shape")

    labels, probabilities = session.run(
        OUTPUT_NAMES,
        {INPUT_NAME: data}
    )

    labels = labels.tolist()

    probs = []
    for p in probabilities:
        probs.append({int(k): float(v) for k, v in p.items()})

    return {
        "label": labels,
        "probabilities": probs
    }
