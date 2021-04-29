import numpy as np
import pandas as pd
import visualization
from joblib import dump, load
from os import path
from string import digits


def partial_results_analysis(models, X_test, y_test):
    visualization.plot_roc_for_all(models, X_test, y_test, np.unique(y_test), n_cols=2)
    visualization.plot_confusion_matrices(models, X_test, y_test, n_cols=2)
    visualization.plot_all()


def results_analysis(best_models, subsets_sizes, X_col, models_dir):
    pd_models = pd.DataFrame(best_models)
    # visualization.show_best_cv_models(pd_models)

    pipeline = load(path.join(models_dir, 'random_forest_' + str(len(X_col)) + '.joblib'))
    rankVar = pd.Series(pipeline.named_steps.clf.feature_importances_ * 100, index=X_col).sort_values(ascending=False)
    visualization.plot_features_info(rankVar, xlabel='Importance Score (%)', title="Features Importance")

    models_names = pd.unique([name.translate({ord(k): None for k in digits})[:-1] for name in pd_models.columns])
    scores_table_per_model = [pd_models[[col for col in pd_models if col.startswith(name)]] for name in models_names]
    scores_table_per_dataset = [pd_models[[col for col in pd_models if col.endswith(fs)]] for fs in subsets_sizes]
    visualization.plot_accuracies(scores_table_per_model, n_cols=3, title='Validation accuracies per Model')
    visualization.plot_accuracies(scores_table_per_dataset, n_cols=2, title='Validation accuracies per Dataset')

    visualization.plot_testing_accuracy(pd_models.transpose()['final_test_score'], models_names, subsets_sizes)

    visualization.plot_all()


def add_test_scores(current_bests, X_test, y_test):
    for name, info in current_bests.items():
        current_bests[name]['final_test_score'] = current_bests[name]['pipeline'].score(X_test, y_test)

    return current_bests
