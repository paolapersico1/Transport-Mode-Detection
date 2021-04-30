import numpy as np
import torch
from torch.utils.data import Dataset

# dataset stored as set of tensors
class TMDDataset(Dataset):
    def __init__(self, X, y):
        self.num_classes = len(np.unique(y))
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    # number of rows
    def __len__(self):
        return self.X.shape[0]

    # function to iterate on observations
    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx]
