import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbn

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
