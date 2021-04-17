import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from torch.utils.data import Dataset
import torch


class MyTochDataset(Dataset):
    def __init__(self, X, y):
        self.num_classes = len(np.unique(y))

        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx]

def loadData():
    data = pd.read_csv(
        'acquisition/dataset_5secondWindow.csv', index_col=0)
    data = data.iloc[:, 4:-1]
    # X_train, y_train, X_test, y_test = train_test_split()
    data = SimpleImputer(strategy='mean').fit_transform(data)
    # print('data shape: {}'.format(data.shape))
    X = data.iloc[:, :-1]
    y = data['target']
    num_classes = len(np.unique(y))

    return X, y, num_classes
