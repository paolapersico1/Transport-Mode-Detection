import pandas as pd
import numpy as np
from data_layer import loadData
from visualization_priori import plot_y
# import seaborn as sb
# import matplotlib.pyplot as plt
# import torch
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix
# from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import RandomForestClassifier
# from sklearn import metrics
# from sklearn.impute import SimpleImputer

if __name__ == '__main__':
    X, y, num_classes = loadData()
    plot_y(y)
    print(X.shape, y.shape, num_classes)
