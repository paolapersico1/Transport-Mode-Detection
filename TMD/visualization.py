from colorsys import hsv_to_rgb
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbn
import pandas as pd
from sklearn.metrics import plot_confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from math import ceil


# function to make the sensor labels more readable
def readable_labels(labels, removePrefix=True, removeSuffix=False):
    if removePrefix:
        labels = list(map(lambda x: x.replace('android.sensor.', ''), labels))
    if removeSuffix:
        labels = [label.split("#", 1)[0] for label in labels]
    return labels


# function to get the hyperparameter from a model pipeline if available
def get_hyperparam(x, hyperparam):
    model_hyperparams = x.named_steps.clf.get_params()
    return model_hyperparams.get(hyperparam, 'n/a')


# function to takes a sensor series as input and returns a dataframe with the stats of each sensor grouped
def group_sensor_features(series):
    sensors = np.unique(readable_labels(series.index, removePrefix=False, removeSuffix=True))
    data = []
    names = ['min', 'max', 'mean', 'std']
    for sensor in sensors:
        data.append([series.get('{}#{}'.format(sensor, statistic), 0) for statistic in names])
    return pd.DataFrame(data, columns=[name.capitalize() for name in names], index=readable_labels(sensors))


# function to show a summary table of the best models returned by validation
def show_best_cv_models(best_models):
    print("\nBest models according to CV:\n")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)

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
                          'Score time (s)': ["{:.2f}".format(x) for x in best_models.loc['mean_score_time']],
                          'Train accuracy': ["{:.2f}".format(x) for x in best_models.loc['mean_train_score']],
                          'Val accuracy': ["{:.2f}".format(x) for x in best_models.loc['mean_test_score']]})
    table.set_index('Model', inplace=True)
    table.sort_values(by=['Val accuracy'], inplace=True, ascending=False)
    print(table)


# function to plot the targets distribution
def plot_class_distribution(y):
    distribution = np.unique(y, return_counts=True)
    fig, axs = plt.subplots(nrows=1, ncols=2)
    step = 1.0 / len(distribution[0])
    colors = [hsv_to_rgb(cur, 0.9, 1) for cur in np.arange(0, 1, step)]
    axs[0].bar(x=distribution[0], height=distribution[1], color=colors)
    axs[1].pie(distribution[1], labels=distribution[0], autopct='%.2f%%', colors=colors)
    fig.suptitle("Number of samples for each class")


def plot_losses(losses):
    plt.figure()
    for fs, loss in losses.items():
        plt.plot(loss, label='{} features'.format(fs.replace('_', '')))
    plt.yscale('log')
    plt.ylabel('Loss value (log scale)')
    plt.xlabel('Epoch')
    plt.title("Loss Progression for Dataset")
    plt.legend()


# function to plot info associated to the sensors
def plot_features_info(series, xlabel, title, operation=np.sum):
    df = group_sensor_features(series)
    ax = df.plot.barh()
    pos1 = ax.get_position()
    ax.set_position([pos1.x0 + 0.06, pos1.y0, pos1.width, pos1.height])
    plt.xlabel(xlabel)
    plt.yticks(np.arange(df.shape[0]),
               labels=["{} ({}%)".format(index, str(round(operation(df.loc[index])))) for index in df.index])
    plt.ylabel("Sensors")
    plt.title(title)


# function to plot the distribution of each sensor feature
def plot_density_all(X, n_measures=4):
    fig, axs = plt.subplots(nrows=int(len(X.columns) / n_measures), ncols=n_measures)
    cols = readable_labels(X.columns)
    for i, col in enumerate(X.columns):
        sbn.kdeplot(data=X, x=col, ax=axs[int(i / n_measures), i % n_measures])
        axs[int(i / n_measures), i % n_measures].set(xticks=[], yticks=[], xlabel='', ylabel='')
        axs[int(i / n_measures), 0].set_ylabel(cols[i].split('#')[0], rotation='horizontal', ha='right')
        axs[0, i % n_measures].set_title(cols[i].split('#')[1])
    fig.suptitle("Distribution per sensor")


# function to plot the roc curve of each model
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
        # one-hot encoded predictions
        one_hot_encoded_preds = label_binarize(model['pipeline'].predict(X), classes=classes)
        fpr = {}
        tpr = {}
        roc_auc = {}
        # for each class
        for i in range(n_classes):
            # false positive rate and true positive rate
            fpr[i], tpr[i], _ = roc_curve(one_hot_encoded_y[:, i], one_hot_encoded_preds[:, i])
            # area under curve
            roc_auc[i] = auc(fpr[i], tpr[i])
        for i, label, color in zip(range(n_classes), classes, colors):
            if n_rows > 1:
                ax = axs[int(j / n_cols), j % n_cols]
            else:
                ax = axs[j % n_cols]
            ax.plot(fpr[i], tpr[i], color=color, lw=lw, label='{0} (area = {1:0.2f})'.format(label, roc_auc[i]))
            # line for comparison purposes
            ax.plot([0, 1], [0, 1], 'k--', lw=lw)
            ax.set(xlim=[0.0, 1.0], ylim=[0.0, 1.05], xlabel='False Positive Rate', ylabel='True Positive Rate',
                   title=' '.join(name.split('_')[:-1]))
            ax.legend(loc="lower right")
    fig.suptitle("ROC Curves per Model (Features Count: {})".format(X.shape[1]))


# function to plot the confusion matrices of each model
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
        ax.set_title(' '.join(name.split('_')[:-1]))
    fig.suptitle("Confusion Matrices per Model (Features Count: {})".format(X.shape[1]))


# one plot for each set of models (grouped by 'dataset size')
def plot_accuracies(scores_table, n_cols=3):
    n_rows = ceil(len(scores_table) / n_cols)
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols)
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    for i, accuracies_table in enumerate(scores_table):
        # sort on validation score
        accuracies_table = accuracies_table.sort_values(by=['mean_test_score'], ascending=False, axis=1)
        X_axis = np.arange(len(accuracies_table.columns))
        if n_rows > 1:
            ax = axs[int(i / n_cols), i % n_cols]
        else:
            ax = axs[i % n_cols]

        # two bars: one for train score and one for validation score
        ax.bar(X_axis - 0.2, accuracies_table.loc['mean_train_score'], 0.4, label='Train Score')
        ax.bar(X_axis + 0.2, accuracies_table.loc['mean_test_score'], 0.4, label='Val Score')

        # show percentages on top
        for p in ax.patches:
            ax.annotate(str(round(p.get_height() * 100)) + "%", (p.get_x() * 1.005, p.get_height() * 1.005))
        ax.legend(loc='lower right')
        plt.sca(ax)
        plt.ylim(0, 1.1)
        plt.xticks(X_axis, [' '.join(x.split('_')[:-1]) for x in accuracies_table.columns], rotation=30)
        ax.set_ylabel("Score")
        ax.set_title('Features Count: {}'.format(accuracies_table.columns[0].split('_')[-1]))
    fig.suptitle('Validation accuracies per Dataset')


# function that takes a series with the models as input and returns a dataframe with models grouped by dataset size
def group_models(series, models_names, subsets_sizes):
    data = [[series[model_name + fs] for fs in subsets_sizes] for model_name in models_names]
    col = [s[1:] + " features" for s in subsets_sizes]
    return pd.DataFrame(data, columns=col, index=models_names)


# function to display one only plot with testing scores
def plot_testing_accuracy(scores_table, models_names, subsets_sizes):
    df = group_models(scores_table, models_names, subsets_sizes)
    ax = df.plot.bar(rot=0)
    plt.xticks(range(len(models_names)), [x.replace('_', ' ') for x in models_names])
    for p in ax.patches:
        ax.annotate(str(round(p.get_height() * 100)) + "%", (p.get_x() * 1.005, p.get_height() * 1.005))
    plt.title('Testing accuracies per Dataset')


# function to show all the plots in the buffer
def plot_all():
    plt.show()
