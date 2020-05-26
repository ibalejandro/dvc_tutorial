import os
import pandas as pd

max_deg = 4
prepared = os.path.join("data", "prepared")
powers = os.path.join("svm", "powers")

if not os.path.exists(powers):
    os.makedirs(powers)

for f in os.listdir(prepared):
    data = pd.read_csv(os.path.join(prepared, f))
    feature_cols = [c for c in data.columns if c != "class"]
    for c in feature_cols:
        for i in range(2, max_deg + 1):
            data[c + str(i)] = data[c] ** i

    data.to_csv(os.path.join(powers, f), index=False)