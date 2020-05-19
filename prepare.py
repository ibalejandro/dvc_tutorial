import os
import numpy as np
import pandas as pd

def split_data(X, t, w=[0.7, 0.15, 0.15]):
    """Takes and array of N x D (N: amount of data points, D: dimension)
       and returns three arrays containing Training, validation and test
       sets
    """
    # normalize train/val/test weights vector
    w = np.array(w)
    w = w/np.sum(w)
    # train/val/test indices
    train_i, val_i, test_i = split_data_indices(X.shape[0], w)
    return X[train_i,], t[train_i], X[val_i,], t[val_i], X[test_i,], t[test_i,]

def split_data_indices(N, w=[0.7, 0.15, 0.15]):
    """Splits the indices [1, .., N] into train, test and validation subsets
    """
    # normalize train/val/test weights vector
    w = np.array(w)
    w = w/np.sum(w)
    # train/val/test indices
    indices = np.random.multinomial(n=1, pvals=w, size=N)==1
    return indices[:,0], indices[:,1], indices[:,2]

data = pd.read_csv("data/iris.data", header=None)
data.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width",
                "class"]
X = data[data.columns[:-1]].values
y = data[["class"]].values

np.random.seed(2)
X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

if not os.path.exists("data/prepared"):
    os.makedirs("data/prepared")

train = np.hstack((X_train, y_train))
train = pd.DataFrame(train, columns=data.columns)
train.to_csv("data/prepared/train.csv", index=False)
val = np.hstack((X_val, y_val))
val = pd.DataFrame(val, columns=data.columns)
val.to_csv("data/prepared/val.csv", index=False)
test = np.hstack((X_test, y_test))
test = pd.DataFrame(train, columns=data.columns)
test.to_csv("data/prepared/test.csv", index=False)
