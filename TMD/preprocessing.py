import numpy as np
from sklearn.impute import SimpleImputer

import visualization
import pandas as pd


def remove_nan(X_train, X_test=None):
    imputer = SimpleImputer(strategy="median").fit(X_train)
    X_train = imputer.transform(X_train)
    if X_test is not None:
        X_test = imputer.transform(X_test)

    return X_train, X_test


def priori_analysis(X, y):
    visualization.plot_class_distribution(y)

    missing_values = [x * 100 / len(X) for x in X.isna().sum()]
    missing_values_series = pd.Series(missing_values, index=X.columns)
    visualization.plot_features_info(missing_values_series, operation=np.mean, xlabel='Missing values (%)',
                                     title="Features missing values")

    visualization.plot_density_all(X)
    visualization.plot_all()

def create_datasets(X):
    # dataset without light, gravity, magnetic, pressure, proximity features
    removable_sensors = ["light", "gravity", "magnetic", "pressure", "proximity"]
    removable_features_1 = [col for col in X.columns if any(sensor in col for sensor in removable_sensors)]

    # dataset with only gyroscope (calibrated and uncalibrated), accelerometer and sound
    relevant_sensors = ["gyroscope", "accelerometer", "sound"]
    removable_features_2 = [col for col in X.columns if all(sensor not in col for sensor in relevant_sensors)]

    X_subsets = [
        X,  # 64 columns
        X.dropna(thresh=(0.7 * X.shape[0]), axis=1),  # 46 columns (kept features with less than 30% missing values)
        X.drop(removable_features_1, axis=1),  # 40 columns
        X.drop(removable_features_2, axis=1)  # 16 columns
    ]

    subsets_sizes = ["_" + str(len(df.columns)) for df in X_subsets]

    return X_subsets, subsets_sizes