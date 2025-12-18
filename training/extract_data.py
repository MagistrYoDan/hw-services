import numpy as np
import pandas as pd
from pathlib import Path

Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)

X = np.random.rand(1000, 10)
y = np.random.randint(0, 2, 1000)

df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
df["target"] = y

df.to_csv("data/raw/data.csv", index=False)
df.to_csv("data/processed/data.csv", index=False)

print("✅ Data extracted")
