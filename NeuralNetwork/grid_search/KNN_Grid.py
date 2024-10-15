import os
import uproot
from sklearn.neighbors import KNeighborsClassifier

############### Import functions from files ###############

from DataPreparation.DataPreparation import load_data, normalize_data, dataset_preparation
from DataPreparation.Evaluation import evaluate_model, evaluate_model_combined, print_plot
from DataPreparation.Grid import perform_grid_search


# Path to ROOT files
file1_path = os.path.join('data', 'toy_sigbkg_categ_offset.root')

treeS_name = "TreeS"
treeB_name = "TreeB"

file1 = uproot.open(file1_path)

treeS = file1[treeS_name]
treeB = file1[treeB_name]

# Upload dataset
X_signal, X_background = load_data(file1_path, treeS_name, treeB_name)

# Normalisation
X_signal_normalized, X_background_normalized = normalize_data(X_signal, X_background)

# Creation train test for full dataset and for cat1 & cat2
(X_train, X_test, y_train, y_test,
 X_train_cat1, X_train_cat2, y_train_cat1, y_train_cat2,
 X_test_cat1, X_test_cat2, y_test_cat1, y_test_cat2) = dataset_preparation(X_signal_normalized, X_background_normalized)



# Parameters for grid search
param_grid = {
    'n_neighbors': [25, 50, 75],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}


# Define the model for the full dataset, for cat1 & cat2
clf_full = KNeighborsClassifier()
clf_cat1 = KNeighborsClassifier()
clf_cat2 = KNeighborsClassifier()

# Grid Search full dataset
best_model_full, best_params_full, best_score_ = perform_grid_search(clf_full, param_grid, X_train, y_train)

# Evaluation of the model on the full dataset
print("Evaluation on Full Dataset:")
tpr, background_rejection = evaluate_model(best_model_full, X_test, y_test) # Return of these variables for comparative ROC plot
print("Best parameter", best_params_full)
print("Best score of Cross-Validation:", best_score_)

# Grid Search cat1 & cat2
best_model_cat1, best_params_cat1, best_score_cat1_ = perform_grid_search(clf_cat1, param_grid, X_train_cat1, y_train_cat1)
best_model_cat2, best_params_cat2, best_score_cat2_ = perform_grid_search(clf_cat2, param_grid, X_train_cat2, y_train_cat2)

# Evaluation of the model on the combined dataset
print("Evaluation on Combined Dataset:")
tpr_combined, background_rejection_combined = evaluate_model_combined(best_model_cat1,best_model_cat2, X_test_cat1, y_test_cat1, X_test_cat2, y_test_cat2) # Return of these variables for comparative ROC plot
print("Best parameter category 1:", best_params_cat1)
print("Best parameter category 2:", best_params_cat2)
print("Best score of Cross-Validation category 1:", best_score_cat1_)
print("Best score of Cross-Validation category 2:", best_score_cat2_)

# Plot comparative ROC curve
print_plot(tpr_combined, background_rejection_combined, tpr, background_rejection)
