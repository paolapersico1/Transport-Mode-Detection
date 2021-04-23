from os import makedirs, path
from functools import reduce
import numpy as np
import pandas as pd
import visualization
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import data_layer
import model_runner
from data_layer import load_data
from sklearn.decomposition import PCA
import torch
from models_params import models
from sklearn.model_selection import train_test_split
from tabulate import tabulate
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from sklearn.impute import SimpleImputer


def pca_analysis():
    std_scaler = StandardScaler()
    smp_imputer = SimpleImputer(strategy="median")
    pca = PCA()
    pca.fit(smp_imputer.fit_transform(std_scaler.fit_transform(X)))
    most_important = [np.abs(pca.components_[i]).argmax() for i in range(X.shape[1])]
    most_important_names = [X.columns[most_important[i]] for i in range(X.shape[1])]
    # visualization.plot_explained_variance(most_important_names, pca.explained_variance_)


def results_analysis(best_estimators):
    # accuracies_train = [x['train_accuracy'] for x in best_estimators.values()]
    # accuracies_test = [x['val_accuracy'] for x in best_estimators.values()]
    # # visualization.plot_confusions(best_estimators, X_trainval, y_trainval)
    # accuracies_train, accuracies_test, models_names = (list(t) for t in zip(*sorted(
    #     zip(accuracies_train, accuracies_test, best_estimators.keys()), reverse=True)))
    # [print(name, '{:.2f}'.format(train_score), '{:.2f}'.format(test_score)) for train_score, test_score, name in
    #  zip(accuracies_train, accuracies_test, models_names)]
    # # visualization.plot_accuracies(models_names, accuracies_train, accuracies_test, False)
    print("okay")

if __name__ == '__main__':
    torch.manual_seed(0)
    torch.set_deterministic(True)
    np.random.seed(0)
    # true aumenta le performance ma lo rende non-deterministico
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    models_dir = 'saved_models'
    use_saved_if_available, save_models = True, True

    if not path.exists(models_dir):
        print("WARNING: Making not existing folder: {}".format(models_dir))
        makedirs(models_dir)

    if not path.exists(path.join(models_dir, "csvs")):
        print("WARNING: Making not existing folder: {}".format(path.join(models_dir, "csvs")))
        makedirs(path.join(models_dir, "csvs"))

    X, y, num_classes = load_data()

    pca_analysis()

    # visualization.plot_class_distribution(y)
    # visualization.plot_missingvalues_var(X)
    # visualization.boxplot(X)

    # visualization.plot_density_all(X)
    # for col in X.columns:
    #     visualization.density(X[col])

    # train with 64,46,40,16 features
    #dataset with features with less than 30% missing values
    X_46 = X.dropna(thresh= (0.7 * X.shape[0]), axis=1)  #46 columns

    #dataset without light, gravity, magnetic, pressure, proximity features
    removable_sensors = ["light", "gravity", "magnetic", "pressure", "proximity"]
    removable_features = [col for col in X.columns if any(sensor in col for sensor in removable_sensors)]
    X_40 = X.drop(removable_features, axis=1)  #40 columns

    #dataset with only gyroscope (calibrated and uncalibrated), accelerometer and sound
    relevant_sensors = ["gyroscope", "accelerometer", "sound"]
    removable_features = [col for col in X.columns if all(sensor not in col for sensor in relevant_sensors)]
    X_16 = X.drop(removable_features, axis=1)  #16 columns

    best_models = {}
    for fs, X in [("", X), ("_46", X_46), ("_40", X_40), ("_16", X_16)]:
        X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
        X_trainval, imputer = data_layer.preprocess(X_trainval)
        X_test = imputer.transform(X_test)

        for est_name, est, params in models:
            est_name = est_name + fs
            file_name = est_name + ".joblib"

            if use_saved_if_available and path.exists(path.join(models_dir, file_name)):
                print("Saved model found: {}".format(est_name))
                best_models[est_name] = {'pipeline': load(path.join(models_dir, file_name))}
                result = pd.read_csv(path.join(models_dir, "csvs", est_name + ".csv"))
            else:
                result, current_pipeline = model_runner.run_trainval(X_trainval, y_trainval, est, params, cv=10)
                best_models[est_name] = {'pipeline': current_pipeline}
                if save_models:
                    result.to_csv(path.join(models_dir, "csvs", est_name + ".csv"))
                    dump(best_models[est_name]['pipeline'], path.join(models_dir, file_name))

            #best_models[est_name]["train_accuracy"] = best_models[est_name]['pipeline'].score(X_trainval, y_trainval)
            best_models[est_name]["train_accuracy"] = result.loc[result['rank_test_score'] == 1]["mean_train_score"].values[0]
            best_models[est_name]["val_accuracy"] = result.loc[result['rank_test_score'] == 1]["mean_test_score"].values[0]
            best_models[est_name]["mean_fit_time"] = result.loc[result['rank_test_score'] == 1]["mean_fit_time"].values[0]
            #best_models[est_name]["predicts"] = best_models[est_name]['pipeline'].predict(X_test)
            #best_models[est_name]["test_accuracy"] = best_models[est_name]['pipeline'].score(X_test, y_test)
        # print('------------------------------------')
        # pd.set_option('display.max_columns', None)
        # pd.set_option('display.max_rows', None)
        # pd.set_option('display.max_colwidth', None)
        # print([result.columns for result in results])
        # names = list(best_estimators.keys())
        # [(print(names[i]), print(
        #     result.loc[:, result.columns.str.startswith("param_")].assign(mean_test_score=result['mean_test_score'],
        #                                                                   rank_test_score=result['rank_test_score'])),
        #   print('---')) for i, result in
        #  enumerate(results)]
    #visualization.plot_roc_for_all(best_models, X_test, y_test, np.unique(y_test))
    results_analysis(best_models)
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

    # preprocess test set with X_trainval means for missing values
