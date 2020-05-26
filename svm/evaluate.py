import sys
import os
import pandas as pd
import numpy as np
import pickle

model_path = sys.argv[1]
data_path = sys.argv[2]
metrics_path = sys.argv[3]

with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)
X = pd.read_csv(data_path)
y = X.pop("class")

y_hat = model.predict(X)
acc = np.mean(y == y_hat)

with open(metrics_path, 'w') as metrics_file:
    metrics_file.write(str(acc))
