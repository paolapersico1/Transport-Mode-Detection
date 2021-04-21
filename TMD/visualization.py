import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbn
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
import pandas as pd


def plot_class_distribution(y):
    plt.bar(x=np.unique(y, return_counts=True)[0], height=np.unique(y, return_counts=True)[1])
    plt.title("Number of samples for each class")
    plt.show()

def plot_missingvalues_var(X):
    x = range(1, X.shape[1]+1)
    y = [x * 100 / len(X) for x in X.isna().sum()]
    plt.barh(y=x, width=y)
    plt.yticks(x, readable_labels(X.columns), size='xx-small')
    for i, v in enumerate(y):
        plt.text(v + 1, i + .25, str(int(v)) + "%", color='blue', size='xx-small')
    plt.title("Percentages of missing values for each feature")
    plt.show()

def boxplot(X):
    x = range(1, X.shape[1] + 1)
    plt.boxplot(X)
    plt.xticks(x, readable_labels(X.columns), size='xx-small', rotation=90)
    plt.show()

def density(X):
    sbn.kdeplot(X)
    plt.show()

def plot_explained_variance(labels, y):
    x = range(1, len(labels)+1)
    plt.bar(x, height=y)
    plt.xticks(x, readable_labels(labels), size='xx-small', rotation=90)
    plt.title("Explained variance for each variable")
    plt.show()

def readable_labels(labels):
    return list(map(lambda x: x.replace('android.sensor.',''), labels))

def plot_roc(model, X_test, y_test):
    metrics.plot_roc_curve(model, X_test, y_test)
    plt.show()

def plot_confusion(model, X, y, title):
    plot_confusion_matrix(model, X, y)
    plt.title(title)
    # plt.show()

def show_best_cv_models(best_models):
    print("Best models according to CV:\n")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)

    table = pd.DataFrame({'Model': best_models.keys(),
                          'Pre-processing': [x['model'].named_steps.scaler for x in best_models.values()],
                          'C': [get_hyperparam(x, "C") for x in best_models.values()],
                          'gamma': [get_hyperparam(x, "gamma") for x in best_models.values()],
                          'degree': [get_hyperparam(x, "degree") for x in best_models.values()],
                          'n_estimators': [get_hyperparam(x, "n_estimators") for x in best_models.values()],
                          'Accuracy': ["{:.2f}".format(x['accuracy']) for x in best_models.values()]})
    table.set_index('Model', inplace=True,)
    table.sort_values(by=['Accuracy'], inplace=True, ascending=False)
    print(table)

def get_hyperparam(x, hyperparam):
    model_hyperparams = x['model'].named_steps.clf.get_params()
    if hyperparam in model_hyperparams.keys():
        result = model_hyperparams[hyperparam]
    else:
        result = "n/a"
    return result
