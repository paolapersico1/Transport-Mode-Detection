import os
import os.path as path
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
from sklearn.discriminant_analysis import  QuadraticDiscriminantAnalysis
import data_layer
import model_runner
import visualization
from data_layer import loadData
from sklearn.decomposition import PCA
import torch
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from tabulate import tabulate
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from sklearn.impute import SimpleImputer

if __name__ == '__main__':
    torch.manual_seed(0)
    #torch.use_deterministic_algorithms(True)
    np.random.seed(0)
    # true aumenta le performance ma lo rende non-deterministico
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    models_dir = 'saved_models'

    X, y, num_classes = loadData()

    removable_sensors = ["light", "gravity", "magnetic", "pressure", "proximity"]
    selected_features = [col for col in X.columns if all(sensor not in col for sensor in removable_sensors)]
    selected_cols = X.columns.get_indexer(selected_features)

    stdScaler = StandardScaler()
    smpImputer = SimpleImputer(strategy="median")
    pca = PCA()
    pca.fit(smpImputer.fit_transform(stdScaler.fit_transform(X)))
    most_important = [np.abs(pca.components_[i]).argmax() for i in range(X.shape[1])]
    most_important_names = [X.columns[most_important[i]] for i in range(X.shape[1])]
    # visualization.plot_explained_variance(most_important_names, pca.explained_variance_)


    #visualization_priori.plot_class_distribution(y)
    #visualization.plot_missingvalues_var(X)
    #visualization.boxplot(X)
    # for col in X.columns:
    #     visualization.density(X[col])

    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    X_trainval, imputer = data_layer.preprocess(X_trainval)

    models = [
        (
            "svc_linear",
            SVC(kernel="linear"),
            {
                'clf__C': np.logspace(-3, 1, 5)
            }
        ),
        (
            "svc_poly",
            SVC(kernel="poly"),
            {
                'clf__C': np.logspace(-3, 1, 5),
                'clf__degree': range(2, 6)
            }
        ),
        (
            "svc_rbf",
            SVC(kernel="rbf"),
            {
                'clf__C': np.logspace(-3, 1, 5),
                'clf__gamma': np.logspace(-3, 1, 5)
            }
        ),
        (
            "gaussian",
            GaussianNB(),
            {}
        ),
        (
            "qda",
            QuadraticDiscriminantAnalysis(),
            {}
        ),
        (
            "random_forest",
            RandomForestClassifier(random_state=42, n_jobs=4),
            {
                'clf__n_estimators': [10, 20, 50, 100, 200, 300]
            }
        )
    ]
    best_models = {}
    for est_name, est, params in models:
        if os.path.exists(path.join(models_dir, est_name + ".joblib")):
            print("Saved model found: {}".format(est_name))
            best_models[est_name] = {'model': load(path.join(models_dir, est_name + ".joblib"))}
        else:
            best_models[est_name] = {'model': model_runner.run_trainval(X_trainval, y_trainval, est, params, selected_cols, cv=10)}
            dump(best_models[est_name]['model'], path.join(models_dir, est_name + ".joblib"))

        best_models[est_name]["accuracy"] = best_models[est_name]['model'].score(X_trainval, y_trainval)
        # visualization.plot_confusion(best_models[est_name]['model'], X_trainval, y_trainval, est_name)

    visualization.show_best_cv_models(best_models)

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

    #preprocess test set with X_trainval means for missing values



