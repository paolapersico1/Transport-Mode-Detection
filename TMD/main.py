import warnings
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from os import makedirs, path

import evaluation
import visualization
import data_layer
import model_runner
import preprocessing
from models_config import models

from pytorch import nn_main


def set_deterministic_behavior():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(0)


if __name__ == '__main__':
    # warnings.filterwarnings("ignore")
    set_deterministic_behavior()

    models_dir = 'saved_models'
    nn_models_dir = path.join('pytorch', 'saved_models')
    use_saved_if_available, save_models = True, False

    if not path.exists(models_dir):
        print("WARNING: Making not existing folder: {}".format(models_dir))
        makedirs(models_dir)
        makedirs(path.join(models_dir, "csvs"))

    if not path.exists(nn_models_dir):
        print("WARNING: Making not existing folder: {}".format(nn_models_dir))
        makedirs(nn_models_dir)
        makedirs(path.join(nn_models_dir, "csvs"))

    X, y, num_classes = data_layer.load_data()
    lenc = LabelEncoder()
    y_encoded = lenc.fit_transform(y)
    # preprocessing.priori_analysis(X, y)
    X_subsets, subsets_sizes = preprocessing.create_datasets(X)

    best_models = {}
    # for each dataset subset
    for fs, X in zip(subsets_sizes, X_subsets):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
        X_train, X_test = preprocessing.remove_nan(X_train, X_test)

        current_bests = model_runner.retrieve_best_models(X_train, y_train, fs, use_saved_if_available, save_models,
                                                          models_dir)
        current_bests = evaluation.add_test_scores(current_bests, X_test, y_test)
        best_models.update(current_bests)
        # plot roc curve and confusion matrix of each model
        # evaluation.partial_results_analysis(current_bests, X_test, y_test)

        best_mlp = nn_main.run(X.to_numpy(), y_encoded, nn_models_dir)
        best_models.update(best_mlp)

    #display cross-validation and testing complete results
    models_names = [est_name for est_name, _, _ in models]
    models_names.append("mlp")
    evaluation.results_analysis(best_models, models_names, subsets_sizes)

