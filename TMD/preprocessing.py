import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import visualization


def remove_nan(X_train, X_test=None):
    imputer = SimpleImputer(strategy="median").fit(X_train)
    X_train = imputer.transform(X_train)
    if X_test is not None:
        X_test = imputer.transform(X_test)

    return X_train, X_test

def pca_analysis(X):
    #fit PCA on standardized dataset without missing values
    std_scaler = StandardScaler()
    pca = PCA()
    pca.fit(remove_nan(std_scaler.fit_transform(X))[0])
    #get most important components names
    most_important = [np.abs(pca.components_[i]).argmax() for i in range(X.shape[1])]
    most_important_names = [X.columns[most_important[i]] for i in range(X.shape[1])]
    #plot the explained variance of each component
    visualization.plot_explained_variance(most_important_names, pca.explained_variance_)

def priori_analysis(X, y):
    pca_analysis(X)
    visualization.plot_class_distribution(y)
    visualization.plot_missingvalues_var(X)
    #visualization.boxplot(X)
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