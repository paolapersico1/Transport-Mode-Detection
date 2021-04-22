import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

def preprocess(X_trainval):
    imputer = SimpleImputer(strategy="median").fit(X_trainval)
    X_trainval = imputer.transform(X_trainval)
    # preprocess_pipeline = Pipeline([('fillnan', SimpleImputer(strategy="median")),
    #                                 ('scaler', StandardScaler())])

    return X_trainval, imputer




