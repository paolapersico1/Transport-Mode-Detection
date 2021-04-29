from colorsys import hsv_to_rgb
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbn
import pandas as pd
from sklearn.metrics import plot_confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from math import ceil


def readable_labels(labels, removePrefix=True, removeSuffix= False):
    if removePrefix:
        labels = list(map(lambda x: x.replace('android.sensor.', ''), labels))
    if removeSuffix:
        labels = [label.split("#", 1)[0] for label in labels]
    return labels


def get_hyperparam(x, hyperparam):
    model_hyperparams = x.named_steps.clf.get_params()
    if hyperparam in model_hyperparams.keys():
        result = model_hyperparams[hyperparam]
    else:
        result = "n/a"
    return result

def group_sensor_features(series):
    sensors = np.unique(readable_labels(series.index, removePrefix=False, removeSuffix=True))
    data = [[series[sensor + "#min"], series[sensor + "#max"], series[sensor + "#mean"],
             series[sensor + "#std"]] for sensor in sensors]
    df = pd.DataFrame(data, columns=["Min", "Max", "Mean", "Std"], index=readable_labels(sensors))
    return df

def show_best_cv_models(best_models):
    print("\nBest models according to CV:\n")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)
    print([x for x in best_models])

    table = pd.DataFrame({'Model': best_models.columns,
                          'Pre-processing': [x.named_steps.scaler if x != "mlp" else "MinMaxScaler()"
                                             for x in best_models.loc['pipeline']],
                          'C': [get_hyperparam(x, "C") if x != "mlp" else "n/a"
                                for x in best_models.loc['pipeline']],
                          'gamma': [get_hyperparam(x, "gamma") if x != "mlp" else "n/a"
                                    for x in best_models.loc['pipeline']],
                          'degree': [get_hyperparam(x, "degree") if x != "mlp" else "n/a"
                                     for x in best_models.loc['pipeline']],
                          'n_estimators': [get_hyperparam(x, "n_estimators") if x != "mlp" else "n/a"
                                           for x in best_models.loc['pipeline']],
                          'hidden_size': [best_models[x].loc["hidden_size"] if best_models[x].loc['pipeline'] == "mlp"
                                          else "n/a" for x in best_models],
                          'epochs': [best_models[x].loc["epochs"] if best_models[x].loc['pipeline'] == "mlp"
                                          else "n/a" for x in best_models],
                          'batch_size': [best_models[x].loc["batch_size"] if best_models[x].loc['pipeline'] == "mlp"
                                     else "n/a" for x in best_models],
                          'decay': [best_models[x].loc["decay"] if best_models[x].loc['pipeline'] == "mlp"
                                     else "n/a" for x in best_models],
                          'Fit time (s)': ["{:.2f}".format(x) for x in best_models.loc['mean_fit_time']],
                          'Train accuracy': ["{:.2f}".format(x) for x in best_models.loc['mean_train_score']],
                          'Val accuracy': ["{:.2f}".format(x) for x in best_models.loc['mean_test_score']]})
    table.set_index('Model', inplace=True)
    table.sort_values(by=['Val accuracy'], inplace=True, ascending=False)
    print(table)


def plot_class_distribution(y):
    distribution = np.unique(y, return_counts=True)
    # count = np.sum(distribution[1])
    # [print(x, '{:.2f}%'.format(y / count * 100)) for x, y in zip(distribution[0], distribution[1])]
    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].bar(x=distribution[0], height=distribution[1])
    axs[1].pie(distribution[1], labels=distribution[0], autopct='%.2f%%', )
    fig.suptitle("Number of samples for each class")

def plot_features_info(series, xlabel, title):
    df = group_sensor_features(series)
    df.plot.barh()
    plt.xlabel(xlabel)
    plt.ylabel("Sensors")
    plt.title(title)


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
    n_rows = ceil(len(models) / n_cols)
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols)
    plt.subplots_adjust(wspace=0.4, hspace=0.3)
    for j, (name, model) in enumerate(models.items()):
        one_hot_encoded_preds = label_binarize(model['pipeline'].predict(X), classes=classes)
        fpr = {}
        tpr = {}
        roc_auc = {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(one_hot_encoded_y[:, i], one_hot_encoded_preds[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        for i, label, color in zip(range(n_classes), classes, colors):
            if n_rows > 1:
                ax = axs[int(j / n_cols), j % n_cols]
            else:
                ax = axs[j % n_cols]
            ax.plot(fpr[i], tpr[i], color=color, lw=lw, label='{0} (area = {1:0.2f})'.format(label, roc_auc[i]))
            ax.plot([0, 1], [0, 1], 'k--', lw=lw)
            ax.set(xlim=[0.0, 1.0], ylim=[0.0, 1.05], xlabel='False Positive Rate', ylabel='True Positive Rate',
                   title=name)
            ax.legend(loc="lower right")
    fig.suptitle("ROC Curves per Model (Dataset Size: {})".format(X.shape[1]))


def plot_confusion_matrices(models, X, y, n_cols=3):
    n_rows = ceil(len(models) / n_cols)
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols)
    plt.subplots_adjust(wspace=0.4, hspace=0.3)
    for i, (name, model) in enumerate(models.items()):
        if n_rows > 1:
            ax = axs[int(i / n_cols), i % n_cols]
        else:
            ax = axs[i % n_cols]
        plot_confusion_matrix(model['pipeline'], X, y, ax=ax)
        ax.set_title(name)
    fig.suptitle("Confusion Matrices per Model (Dataset Size: {})".format(X.shape[1]))


def plot_accuracies(scores_table, n_cols=3, title="", testing=False):
    n_rows = ceil(len(scores_table) / n_cols)
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols)
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    for i, accuracies_table in enumerate(scores_table):
        accuracies_table = accuracies_table.sort_values(by=['final_test_score' if testing else 'mean_test_score'],
                                                        ascending=False, axis=1)
        X_axis = np.arange(len(accuracies_table.columns))
        if n_rows > 1:
            ax = axs[int(i / n_cols), i % n_cols]
        else:
            ax = axs[i % n_cols]
        if testing:
            bars = ax.bar(X_axis + 0.2, accuracies_table.loc['final_test_score'], 0.4, label='Test Score')
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + 0.1, yval + 0.01, str(int(yval * 100)) + "%", size='xx-small')
        else:
            ax.bar(X_axis - 0.2, accuracies_table.loc['mean_train_score'], 0.4, label='Train Score')
            ax.bar(X_axis + 0.2, accuracies_table.loc['mean_test_score'], 0.4, label='Val Score')
            ax.legend()

        plt.sca(ax)
        plt.ylim(0, 1.1)
        plt.xticks(X_axis, accuracies_table.columns, rotation=30)
        ax.set_ylabel("Score")
    fig.suptitle(title)


def plot_all():
    plt.show()
