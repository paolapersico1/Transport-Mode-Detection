import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
from sklearn.impute import SimpleImputer


class MyTorchDataset(Dataset):
    def __init__(self, X, y):
        self.num_classes = len(np.unique(y))

        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx]

def loadData():
    data = pd.read_csv('acquisition/dataset_5secondWindow.csv', index_col=0)
    data = data.iloc[:, 4:-1]
    # print('data shape: {}'.format(data.shape))
    X = data.iloc[:, :-1]
    y = data['target']
    num_classes = len(np.unique(y))

    return X, y, num_classes

def preprocess(X_train, X_val, method="std"):
    X_train_mean = np.mean(X_train, axis=0)
    X_train_std = np.std(X_train, axis=0)
    X_train_median = np.median(X_train, axis=0)

    X_train = X_train.fillna(X_train_median)
    X_val = X_val.fillna(X_train_median)


    if method == "std":
        X_train = (X_train - X_train_mean) / X_train_std
        X_val = (X_val - X_train_mean) / X_train_std
    elif method == "scaling":
        X_train_min = np.min(X_train, axis=0)
        X_train_max = np.max(X_train, axis=0)
        X_train_range = X_train_max - X_train_min
        X_train = (X_train - X_train_min) / X_train_range
        X_val = (X_train - X_train_min) / X_train_range
    else:
        raise ValueError("Method not supported")

    return X_train, X_val



