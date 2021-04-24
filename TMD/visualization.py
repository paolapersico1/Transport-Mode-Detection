from colorsys import hsv_to_rgb
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbn
import pandas as pd
from sklearn.metrics import plot_confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from math import ceil


def readable_labels(labels):
    return list(map(lambda x: x.replace('android.sensor.', ''), labels))


def get_hyperparam(x, hyperparam):
    model_hyperparams = x['pipeline'].named_steps.clf.get_params()
    if hyperparam in model_hyperparams.keys():
        result = model_hyperparams[hyperparam]
    else:
        result = "n/a"
    return result


def show_best_cv_models(best_models):
    print("\nBest models according to CV:\n")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)

    table = pd.DataFrame({'Model': best_models.keys(),
                          'Pre-processing': [x['pipeline'].named_steps.scaler for x in best_models.values()],
                          'C': [get_hyperparam(x, "C") for x in best_models.values()],
                          'gamma': [get_hyperparam(x, "gamma") for x in best_models.values()],
                          'degree': [get_hyperparam(x, "degree") for x in best_models.values()],
                          'n_estimators': [get_hyperparam(x, "n_estimators") for x in best_models.values()],
                          'Fit time (s)': ["{:.2f}".format(x['mean_fit_time']) for x in best_models.values()],
                          'Train accuracy': ["{:.2f}".format(x['train_accuracy']) for x in best_models.values()],
                          'Val accuracy': ["{:.2f}".format(x['val_accuracy']) for x in best_models.values()]})
    table.set_index('Model', inplace=True, )
    table.sort_values(by=['Val accuracy'], inplace=True, ascending=False)
    print(table)


def plot_class_distribution(y):
    distribution = np.unique(y, return_counts=True)
    count = np.sum(distribution[1])
    [print(x, '{:.2f}%'.format(y / count * 100)) for x, y in zip(distribution[0], distribution[1])]
    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].bar(x=distribution[0], height=distribution[1])
    axs[1].pie(distribution[1], labels=distribution[0], autopct='%.2f%%', )
    fig.suptitle("Number of samples for each class")


def plot_missingvalues_var(X):
    plt.figure()
    x = range(1, X.shape[1] + 1)
    y = [x * 100 / len(X) for x in X.isna().sum()]
    plt.barh(y=x, width=y)
    plt.yticks(x, readable_labels(X.columns), size='xx-small')
    for i, v in enumerate(y):
        plt.text(v + 1, i + .25, str(int(v)) + "%", color='blue', size='xx-small')
    plt.title("Percentages of missing values for each feature")


def plot_density_all(X, n_measures=4):
    fig, axs = plt.subplots(nrows=int(len(X.columns) / n_measures), ncols=n_measures)
    cols = readable_labels(X.columns)
    for i, col in enumerate(X.columns):
        sbn.kdeplot(data=X, x=col, ax=axs[int(i / n_measures), i % n_measures])
        axs[int(i / n_measures), i % n_measures].set(xticks=[], yticks=[], xlabel='', ylabel='')
        axs[int(i / n_measures), 0].set_ylabel(cols[i].split('#')[0], rotation='horizontal', ha='right')
        axs[0, i % n_measures].set_title(cols[i].split('#')[1])
    fig.suptitle("Distribution per sensor")


def plot_explained_variance(labels, y):
    plt.figure()
    x = range(1, len(labels) + 1)
    plt.barh(x, width=y)
    plt.yticks(x, readable_labels(labels), size='xx-small')
    plt.title("Explained variance for each variable")


def plot_roc_for_all(models, X, y, classes, n_cols=3):
    n_classes = len(classes)
    lw = 2
    one_hot_encoded_y = label_binarize(y, classes=classes)
    step = 1.0 / n_classes
    colors = [hsv_to_rgb(cur, 1, 1) for cur in np.arange(0, 1, step)]
    fig, axs = plt.subplots(nrows=ceil(len(models) / n_cols), ncols=n_cols)
    plt.subplots_adjust(hspace=0.25)
    for j, (name, model) in enumerate(models.items()):
        one_hot_encoded_preds = label_binarize(model['pipeline'].predict(X), classes=classes)
        fpr = {}
        tpr = {}
        roc_auc = {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(one_hot_encoded_y[:, i], one_hot_encoded_preds[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        for i, label, color in zip(range(n_classes), classes, colors):
            axs[int(j / n_cols), j % n_cols].plot(fpr[i], tpr[i], color=color, lw=lw,
                                                  label='{0} (area = {1:0.2f})'
                                                        ''.format(label, roc_auc[i]))
            axs[int(j / n_cols), j % n_cols].plot([0, 1], [0, 1], 'k--', lw=lw)
            axs[int(j / n_cols), j % n_cols].set(xlim=[0.0, 1.0], ylim=[0.0, 1.05], xlabel='False Positive Rate',
                                                 ylabel='True Positive Rate', title=name)
            axs[int(j / n_cols), j % n_cols].legend(loc="lower right")
    fig.suptitle("ROC Curves per Model (Dataset Size: {})".format(X.shape[1]))


def plot_confusions(models, X, y, n_cols=3):
    fig, axs = plt.subplots(nrows=ceil(len(models) / n_cols), ncols=n_cols)
    plt.subplots_adjust(hspace=0.25)
    for i, (name, model) in enumerate(models.items()):
        plot_confusion_matrix(model['pipeline'], X, y, ax=axs[int(i / n_cols), i % n_cols])
        axs[int(i / n_cols), i % n_cols].set_title(name)
    fig.suptitle("Confusion Matrices per Model (Dataset Size: {})".format(X.shape[1]))


def plot_accuracies(accuracies_table_per_models, n_cols=3, title=""):
    fig, axs = plt.subplots(nrows=ceil(len(accuracies_table_per_models) / n_cols), ncols=n_cols)
    plt.subplots_adjust(hspace=0.25)
    for i, accuracies_table in enumerate(accuracies_table_per_models):
        accuracies_table = accuracies_table.sort_values(by=['val_accuracy'], ascending=False, axis=1)
        X_axis = np.arange(len(accuracies_table.columns))
        axs[int(i / n_cols), i % n_cols].bar(X_axis - 0.2, accuracies_table.loc['train_accuracy'], 0.4,
                                             label='Train Score')
        axs[int(i / n_cols), i % n_cols].bar(X_axis + 0.2, accuracies_table.loc['val_accuracy'], 0.4,
                                             label='Val Score')
        plt.sca(axs[int(i / n_cols), i % n_cols])
        plt.xticks(X_axis, accuracies_table.columns, rotation=30)
        axs[int(i / n_cols), i % n_cols].set_ylabel("Score")
        axs[int(i / n_cols), i % n_cols].legend()
    fig.suptitle(title)


def plot_all():
    plt.show()
