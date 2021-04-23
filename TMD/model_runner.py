import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def run_trainval(X_trainval, y_trainval, clf, params, cv=5, verbose=True):
    params["scaler"] = [StandardScaler(), MinMaxScaler()]
    pipeline = Pipeline([('scaler', StandardScaler()), ('clf', clf)])

    grid_search = GridSearchCV(pipeline, params, cv=cv, verbose=10 if verbose else 0, n_jobs=16, return_train_score=True)
    grid_search.fit(X_trainval, y_trainval)

    return pd.DataFrame(grid_search.cv_results_), grid_search.best_estimator_
