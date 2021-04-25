import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
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


def load_data():
    data = pd.read_csv('dataset/dataset_5secondWindow.csv', index_col=0)
    data = data.iloc[:, 4:-1]
    # print('data shape: {}'.format(data.shape))
    X = data.iloc[:, :-1]
    y = data['target']
    num_classes = len(np.unique(y))

    return X, y, num_classes
