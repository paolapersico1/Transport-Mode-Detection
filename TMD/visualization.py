from colorsys import hsv_to_rgb
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbn
import pandas as pd
from sklearn.metrics import plot_confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from math import ceil


def readable_labels(labels, removePrefix=True, removeSuffix=False):
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
    data = [[series.get(sensor + "#min", 0), series.get(sensor + "#max", 0), series.get(sensor + "#mean", 0),
             series.get(sensor + "#std", 0)] for sensor in sensors]
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
    colors = ["red", "yellow", "blue", "green", "orange"]
    axs[0].bar(x=distribution[0], height=distribution[1], color=colors)
    axs[1].pie(distribution[1], labels=distribution[0], autopct='%.2f%%', colors=colors)
    fig.suptitle("Number of samples for each class")

def plot_features_info(series, xlabel, title, operation=np.sum):
    # df = group_sensor_features(series) if group else pd.DataFrame(series.rename(lambda x: x.replace('android.sensor.', '')))
    df = group_sensor_features(series)
    ax = df.plot.barh()
    pos1 = ax.get_position()
    ax.set_position([pos1.x0 + 0.06, pos1.y0, pos1.width, pos1.height])
    plt.xlabel(xlabel)
    plt.yticks(np.arange(df.shape[0]), labels=["{} ({}%)".format(index, str(round(operation(df.loc[index])))) for index in df.index])
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


def plot_accuracies(scores_table, n_cols=3, title=""):
    n_rows = ceil(len(scores_table) / n_cols)
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols)
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    for i, accuracies_table in enumerate(scores_table):
        accuracies_table = accuracies_table.sort_values(by=['mean_test_score'], ascending=False, axis=1)
        X_axis = np.arange(len(accuracies_table.columns))
        if n_rows > 1:
            ax = axs[int(i / n_cols), i % n_cols]
        else:
            ax = axs[i % n_cols]

        ax.bar(X_axis - 0.2, accuracies_table.loc['mean_train_score'], 0.4, label='Train Score')
        ax.bar(X_axis + 0.2, accuracies_table.loc['mean_test_score'], 0.4, label='Val Score')
        for p in ax.patches:
            ax.annotate(str(round(p.get_height() * 100)) + "%", (p.get_x() * 1.005, p.get_height() * 1.005))
        ax.legend(loc='lower right')
        plt.sca(ax)
        plt.ylim(0, 1.1)
        plt.xticks(X_axis, accuracies_table.columns, rotation=30)
        ax.set_ylabel("Score")
    fig.suptitle(title)

def group_models(series, models_names, subsets_sizes):
    data = [[series[model_name + fs] for fs in subsets_sizes] for model_name in models_names]
    col = [s[1:] + " features" for s in subsets_sizes]
    df = pd.DataFrame(data, columns=col, index=models_names)
    return df

def plot_testing_accuracy(scores_table, models_names, subsets_sizes):
    df = group_models(scores_table, models_names, subsets_sizes)
    ax = df.plot.bar(rot=0)
    for p in ax.patches:
        ax.annotate(str(round(p.get_height() * 100)) + "%", (p.get_x() * 1.005, p.get_height() * 1.005))
    plt.title('Testing accuracies per Dataset')


def plot_all():
    plt.show()
