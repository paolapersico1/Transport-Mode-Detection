import numpy as np
import pandas as pd
import visualization

def partial_results_analysis(models, X_test, y_test):
    visualization.plot_roc_for_all(models, X_test, y_test, np.unique(y_test))
    visualization.plot_confusion_matrices(models, X_test, y_test)
    visualization.plot_all()

def results_analysis(best_models, models_names, subsets_sizes):
    pd_models = pd.DataFrame(best_models)
    visualization.show_best_cv_models(pd_models)

    scores_table_per_model = [pd_models[[col for col in pd_models if col.startswith(name)]] for name in models_names]
    scores_table_per_dataset = [pd_models[[col for col in pd_models if col.endswith(fs)]] for fs in subsets_sizes]
    visualization.plot_accuracies(scores_table_per_model, title='Cross-validation accuracies per Model')
    visualization.plot_accuracies(scores_table_per_dataset, n_cols=2, title='Cross-validation accuracies per Dataset')
    visualization.plot_accuracies(scores_table_per_dataset, n_cols=2, title='Testing accuracies per Dataset', testing=True)
    visualization.plot_all()

def add_test_scores(current_bests, X_test, y_test):
    for name, info in current_bests.items():
        current_bests[name]['final_test_score'] = current_bests[name]['pipeline'].score(X_test, y_test)

    return current_bests