import pandas as pd
import pickle
from pathlib import Path

df = pd.read_csv("data/processed/data.csv")
X = df.drop("target", axis=1)

with open("artifacts/model.pkl", "rb") as f:
    model = pickle.load(f)

preds = model.predict(X)

Path("artifacts").mkdir(exist_ok=True)
pd.DataFrame({"prediction": preds}).to_csv(
    "artifacts/predictions.csv", index=False
)

print("✅ Inference done")
