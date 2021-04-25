import warnings
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

import evaluation
import visualization
import data_layer
import model_runner
import preprocessing
from models_config import models

def set_deterministic_behavior():
    torch.manual_seed(0)
    torch.set_deterministic(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(0)

if __name__ == '__main__':
    #warnings.filterwarnings("ignore")
    set_deterministic_behavior()

    models_dir = 'saved_models'
    use_saved_if_available, save_models = True, False

    X, y, num_classes = data_layer.load_data()
    #preprocessing.priori_analysis(X, y)
    X_subsets, subsets_sizes = preprocessing.create_datasets(X)

    best_models = {}
    #for each dataset subset
    for fs, X in zip(subsets_sizes, X_subsets):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
        X_train, X_test = preprocessing.remove_nan(X_train, X_test)

        current_bests = model_runner.retrieve_best_models(X_train, y_train, fs, use_saved_if_available, save_models, models_dir)
        current_bests = evaluation.add_test_scores(current_bests, X_test, y_test)
        best_models.update(current_bests)
        # plot roc curve and confusion matrix of each model
        evaluation.partial_results_analysis(current_bests, X_test, y_test)


    # display cross-validation and testing complete results
    models_names = [est_name for est_name, _, _ in models]
    evaluation.results_analysis(best_models, models_names, subsets_sizes)


    # kf = KFold(n_splits=10)
    # for train_index, val_index in kf.split(X_trainval):
    #     X_train, X_val, y_train, y_val = X_trainval.iloc[train_index], X_trainval.iloc[val_index],\
    #                                      y_trainval.iloc[train_index], y_trainval.iloc[val_index]
    #
    #     models = [SVM,...]
    #     methods = ["std", "scaling"]
    #     scores = np.array((2,5))
    #
    #     for i in range(len(methods)):
    #         X_train, X_val = data_layer.preprocess(X_train, X_val, method=methods[i])
    #         for j in range(len(models)):
    #             #scores[i,j] = models[j].train_model(X_train, X_val)
    #             from sklearn.svm import SVC
    #
    #             param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
    #                           'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
    #
    #
    #
    #
    #
    #     break

    # preprocess test set with X_trainval means for missing values
