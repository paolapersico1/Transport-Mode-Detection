from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import FunctionTransformer

import data_layer


def run_trainval(X_trainval, y_trainval, clf, params, selected_cols, cv=5, verbose=True):
    params["scaler"] = [StandardScaler(), MinMaxScaler()]
    params["feature_selection"] = [data_layer.FeatureSelection(),
                                   data_layer.FeatureSelection(selected_cols)]
    pipeline = Pipeline([('feature_selection', data_layer.FeatureSelection([])), ('scaler', StandardScaler()), ('clf', clf)])

    try:
        grid_search = GridSearchCV(pipeline, params, cv=cv, verbose=10 if verbose else 0, n_jobs=8)
        grid_search.fit(X_trainval, y_trainval)

        if verbose:
            print(pd.DataFrame(grid_search.cv_results_))
            # results = pd.DataFrame(grid_search.cv_results_)
            # results['param_clf'] = results["param_clf"].apply(lambda x: str(type(x)))
            # pd.set_option("display.max_colwidth", None)
            # print(results.iloc[results.groupby('param_clf')['mean_test_score'].idxmax()]["params"])
            # print(results[results[''], :])
    except ValueError as E:
        print(pipeline.get_params().keys())
        raise E
    # print(time.time() - time0)

    return grid_search.best_estimator_