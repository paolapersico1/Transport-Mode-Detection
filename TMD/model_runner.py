from joblib import dump, load
from os import makedirs, path
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from models_config import models


def get_rank1_info(result, attribute):
    return result.loc[result['rank_test_score'] == 1][attribute].values[0]


def retrieve_best_models(X_train, y_train, fs, use_saved_if_available, save_models, models_dir):
    best_models = {}

    for est_name, est, params in models:
        est_name = est_name + fs
        file_name = est_name + ".joblib"

        if use_saved_if_available and path.exists(path.join(models_dir, file_name)):
            print("Saved model found: {}".format(est_name))
            best_models[est_name] = {'pipeline': load(path.join(models_dir, file_name))}
            result = pd.read_csv(path.join(models_dir, "csvs", est_name + ".csv"))
        else:
            result, current_pipeline = run_crossvalidation(X_train, y_train, est, params, cv=10)
            best_models[est_name] = {'pipeline': current_pipeline}
            if save_models:
                if not path.exists(models_dir):
                    print("WARNING: Making not existing folder: {}".format(models_dir))
                    makedirs(models_dir)
                    makedirs(path.join(models_dir, "csvs"))

                dump(best_models[est_name]['pipeline'], path.join(models_dir, file_name))
                result.to_csv(path.join(models_dir, "csvs", est_name + ".csv"))

        attributes = ["mean_train_score", "mean_test_score", "mean_fit_time"]
        for attribute in attributes:
            best_models[est_name][attribute] = get_rank1_info(result, attribute)

    return best_models


def run_crossvalidation(X_trainval, y_trainval, clf, params, cv=5, verbose=True):
    params["scaler"] = [StandardScaler(), MinMaxScaler()]
    pipeline = Pipeline([('scaler', StandardScaler()), ('clf', clf)])

    grid_search = GridSearchCV(pipeline, params, cv=cv, verbose=10 if verbose else 0, n_jobs=16,
                               return_train_score=True)
    grid_search.fit(X_trainval, y_trainval)

    return pd.DataFrame(grid_search.cv_results_), grid_search.best_estimator_
