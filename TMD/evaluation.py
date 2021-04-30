import numpy as np
import pandas as pd
import visualization
from string import digits

# function called for each dataset
def partial_results_analysis(models, X_test, y_test, X_cols):
    visualization.plot_roc_for_all(models, X_test, y_test, np.unique(y_test), n_cols=2)
    visualization.plot_confusion_matrices(models, X_test, y_test, n_cols=2)

    # display the importance of each feature based on Random Forest model
    rf = models["random_forest_" + str(len(X_cols))]["pipeline"].named_steps.clf
    rankVar = pd.Series(rf.feature_importances_ * 100, index=X_cols).sort_values(ascending=False)
    visualization.plot_features_info(rankVar, xlabel='Importance Score (%)',
                                     title="Features Importance (Dataset Size: {})".format(len(X_cols)))
    visualization.plot_all()

# function to show the general results
def results_analysis(best_models, subsets_sizes, losses):
    pd_models = pd.DataFrame(best_models)
    models_names = pd.unique([name.translate({ord(k): None for k in digits})[:-1] for name in pd_models.columns])

    # show table with cumulative validation info
    visualization.show_best_cv_models(pd_models)

    # plot validation scores grouped by model type and dataset
    scores_table_per_model = [pd_models[[col for col in pd_models if col.startswith(name)]] for name in models_names]
    scores_table_per_dataset = [pd_models[[col for col in pd_models if col.endswith(fs)]] for fs in subsets_sizes]
    visualization.plot_accuracies(scores_table_per_model, n_cols=3, title='Validation accuracies per Model')
    visualization.plot_accuracies(scores_table_per_dataset, n_cols=2, title='Validation accuracies per Dataset')

    # plot testing score of each model
    visualization.plot_testing_accuracy(pd_models.transpose()['final_test_score'], models_names, subsets_sizes)
    if len(losses.keys()) > 0:
        visualization.plot_losses(losses)
    visualization.plot_all()

# function called by main to add the testing scores to the models metadata
def add_test_scores(current_bests, X_test, y_test):
    for name, info in current_bests.items():
        current_bests[name]['final_test_score'] = current_bests[name]['pipeline'].score(X_test, y_test)

    return current_bests
