import numpy as np
import pandas as pd
import visualization

def testing_results_analysis(models, X_test, y_test):
    visualization.plot_roc_for_all(models, X_test, y_test, np.unique(y_test))
    visualization.plot_confusion_matrices(models, X_test, y_test)
    visualization.plot_all()

def validation_results_analysis(best_models, models_names, subsets_sizes):
    pd_models = pd.DataFrame(best_models)
    visualization.show_best_cv_models(pd_models)

    scores_table_per_model = [pd_models[[col for col in pd_models if col.startswith(name)]] for name in models_names]
    scores_table_per_dataset = [pd_models[[col for col in pd_models if col.endswith(fs)]] for fs in subsets_sizes]
    visualization.plot_accuracies(scores_table_per_model, title='Accuracies per Model')
    visualization.plot_accuracies(scores_table_per_dataset, n_cols=2, title='Accuracies per Dataset Size')
    visualization.plot_all()