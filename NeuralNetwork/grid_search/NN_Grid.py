import os
import uproot
from scikeras.wrappers import KerasClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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

# Seleziona le features e il target separatamente per segnale e sfondo# Upload dataset
X_signal, X_background = load_data(file1_path, treeS_name, treeB_name)

# Normalisation
X_signal_normalized, X_background_normalized = normalize_data(X_signal, X_background)

# Creation train test for full dataset and for cat1 & cat2
(X_train, X_test, y_train, y_test,
 X_train_cat1, X_train_cat2, y_train_cat1, y_train_cat2,
 X_test_cat1, X_test_cat2, y_test_cat1, y_test_cat2) = dataset_preparation(X_signal_normalized, X_background_normalized)

# Parameters for grid search
param_grid = {
    'neurons': [32, 64, 128],
    'drop_out': [0.2,0.3,0.4],
    'learning_rate': [0.001, 0.01, 0.1]
}

# Create the model
def create_model(X_train, neurons=64, drop_out=0.3, learning_rate=0.001):
    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(neurons, activation='relu'),
        layers.Dense(neurons, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Model definitions
keras_model_full = KerasClassifier(model=create_model,drop_out=0.3, learning_rate=0.01, neurons=32, X_train=X_train, verbose=0)
keras_model_cat1 = KerasClassifier(model=create_model,drop_out=0.3, learning_rate=0.01, neurons=32, X_train=X_train_cat1, verbose=0)
keras_model_cat2 = KerasClassifier(model=create_model,drop_out=0.3, learning_rate=0.01, neurons=32, X_train=X_train_cat2, verbose=0)

# Grid Search full dataset
best_model_full, best_params_full, best_score_ = perform_grid_search(keras_model_full, param_grid, X_train, y_train)

# Evaluation of the model on the full dataset
print("Evaluation on Full Dataset:")
tpr, background_rejection = evaluate_model(best_model_full, X_test, y_test) # Return of these variables for comparative ROC plot
print("Best parameter", best_params_full)
print("Best score of Cross-Validation:", best_score_)

# Grid Search cat1 & cat2
best_model_cat1, best_params_cat1, best_score_cat1_ = perform_grid_search(keras_model_cat1, param_grid, X_train_cat1, y_train_cat1)
best_model_cat2, best_params_cat2, best_score_cat2_ = perform_grid_search(keras_model_cat2, param_grid, X_train_cat2, y_train_cat2)

# Evaluation of the model on the combined dataset
print("Evaluation on Combined Dataset:")
tpr_combined, background_rejection_combined = evaluate_model_combined(best_model_cat1,best_model_cat2, X_test_cat1, y_test_cat1, X_test_cat2, y_test_cat2) # Return of these variables for comparative ROC plot
print("Best parameter category 1:", best_params_cat1)
print("Best parameter category 2:", best_params_cat2)
print("Best score of Cross-Validation category 1:", best_score_cat1_)
print("Best score of Cross-Validation category 2:", best_score_cat2_)

# Plot comparative ROC curve
print_plot(tpr_combined, background_rejection_combined, tpr, background_rejection)
