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

if __name__ == '__main__':
    torch.manual_seed(0)
    #torch.use_deterministic_algorithms(True)
    np.random.seed(0)
    # true aumenta le performance ma lo rende non-deterministico
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    models_dir = 'saved_models'

    X, y, num_classes = loadData()

    unused_features = ["android.sensor.light#mean","android.sensor.light#min","android.sensor.light#max","android.sensor.light#std",
                        "android.sensor.gravity#mean","android.sensor.gravity#min","android.sensor.gravity#max","android.sensor.gravity#std",
                        "android.sensor.magnetic_field#mean","android.sensor.magnetic_field#min","android.sensor.magnetic_field#max","android.sensor.magnetic_field#std",
                        "android.sensor.magnetic_field_uncalibrated#mean","android.sensor.magnetic_field_uncalibrated#min","android.sensor.magnetic_field_uncalibrated#max","android.sensor.magnetic_field_uncalibrated#std",
                        "android.sensor.pressure#mean","android.sensor.pressure#min","android.sensor.pressure#max","android.sensor.pressure#std",
                        "android.sensor.proximity#mean","android.sensor.proximity#min","android.sensor.proximity#max","android.sensor.proximity#std"]
    X.drop(unused_features, axis=1)

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
    # X_trainval, imputer = data_layer.preprocess(X_trainval, MinMaxScaler())

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
    best_estimators = {}
    for est_name, est, params in models:
        if os.path.exists(path.join(models_dir, est_name + ".joblib")):
            print("Saved model found: {}".format(est_name))
            best_estimators[est_name] = load(path.join(models_dir, est_name + ".joblib"))
        else:
            best_estimators[est_name] = model_runner.run_trainval(X_trainval, y_trainval, est, params, cv=10)
            dump(best_estimators[est_name], path.join(models_dir, est_name + ".joblib"))
        print("Test set accuracy: {:.2f}".format(best_estimators[est_name].score(X_trainval, y_trainval)))
        visualization.plot_confusion(best_estimators[est_name], X_trainval, y_trainval, est_name)

    print(best_estimators)
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



