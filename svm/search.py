import sys
import os
import pandas as pd
import numpy as np

from sklearn.svm import SVC
import pickle

inpath = sys.argv[1]
outpath = sys.argv[2]

X_train = pd.read_csv(os.path.join(inpath, "train.csv"))
y_train = X_train.pop("class")
X_val = pd.read_csv(os.path.join(inpath, "val.csv"))
y_val = X_val.pop("class")
X_test = pd.read_csv(os.path.join(inpath, "test.csv"))
y_test = X_test.pop("class")

# train SVM at several hyperparametrizations
kernels = ["linear"]
Cs = [10, 40]
best_val_acc = 0
best_model = None

for kernel in kernels:
    for C in Cs:
        svm = SVC(C=C, kernel=kernel, gamma="auto")
        svm.fit(X_train, y_train)

        y_val_hat = svm.predict(X_val)
        val_acc = np.mean(y_val_hat == y_val)

        if val_acc > best_val_acc:
            best_model = svm
            best_val_acc = val_acc

dirname = os.path.dirname(outpath)
os.makedirs(dirname, exist_ok=True)

with open(outpath, "wb") as outfile:
    pickle.dump(svm, outfile)