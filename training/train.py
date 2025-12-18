import pandas as pd
import json
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/data.csv")

X = df.drop("target", axis=1)
y = df["target"]

model = RandomForestClassifier(
    n_estimators=50,
    random_state=42,
    n_jobs=1
)

model.fit(X, y)

preds = model.predict(X)
acc = accuracy_score(y, preds)

Path("artifacts").mkdir(exist_ok=True)
Path("plots").mkdir(exist_ok=True)

with open("artifacts/metrics.json", "w") as f:
    json.dump({"accuracy": acc}, f, indent=2)

with open("artifacts/model.pkl", "wb") as f:
    pickle.dump(model, f)

plt.bar(X.columns, model.feature_importances_)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/feature_importance.png")

print("✅ Model trained")
