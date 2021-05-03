import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC

models = [
   (
       "svc_linear",
       SVC(kernel="linear",probability=True),
       {
           'scaler': [StandardScaler(), MinMaxScaler()],
           'clf__C': np.logspace(1, 3, 5, dtype=np.float32)
       }
   ),
   (
       "svc_poly",
       SVC(kernel="poly",probability=True),
       {
           'scaler': [StandardScaler(), MinMaxScaler()],
           'clf__C': np.logspace(1, 3, 5, dtype=np.float32),
           'clf__degree': range(2, 5)
       }
   ),
   (
       "svc_rbf",
       SVC(kernel="rbf",probability=True),
       {
           'scaler': [StandardScaler(), MinMaxScaler()],
           'clf__C': np.logspace(0, 2, 5, dtype=np.float32),
           'clf__gamma': np.logspace(-2, 2, 5, dtype=np.float32)
       }
   ),
    (
        "Gaussian_NB",
        GaussianNB(),
        {
            'scaler': [StandardScaler(), MinMaxScaler()]
        }
    ),
    (
        "QDA",
        QuadraticDiscriminantAnalysis(),
        {
            'scaler': [StandardScaler(), MinMaxScaler()]
        }
    ),
    (
        "Random_Forest",
        RandomForestClassifier(random_state=42, n_jobs=8),
        {
            'scaler': [StandardScaler(), MinMaxScaler()],
            # 'clf__criterion': ['gini', 'entropy'] # since gini works well, we don't need to check entropy
            'clf__n_estimators': [10, 20, 50, 100, 200, 300]
        }
    )
]
