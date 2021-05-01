import pandas as pd
import numpy as np


# function to load the dataset
def load_data():
    data = pd.read_csv('dataset/dataset_5secondWindow.csv', index_col=0)
    # remove the first 4 columns and the last one
    data = data.iloc[:, 4:-1]
    X = data.iloc[:, :-1]
    y = data['target']
    num_classes = len(np.unique(y))

    return X, y, num_classes
