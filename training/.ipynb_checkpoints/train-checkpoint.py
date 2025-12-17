import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
from pathlib import Path

X = np.random.rand(1000, 10)
y = np.random.randint(0, 2, 1000)

model = RandomForestClassifier(
    n_estimators=50,
    random_state=42,
    n_jobs=1
)
model.fit(X, y)

Path("artifacts").mkdir(exist_ok=True)
with open("artifacts/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Модель обучена и сохранена")