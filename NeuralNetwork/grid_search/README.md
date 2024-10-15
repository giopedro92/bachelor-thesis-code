# grid_search folder code
This directory comprises four distinct files (`BDT_Grid.py`, `KNN_Grid.py`, `Random_Forest_Grid.py`, `NN_Grid.py`) alongside a subdirectory named DataPreparation. Within the `DataPreparation` directory reside essential function files instrumental in the grid search scripts: `DataPreparation.py`, `Evaluation.py`, `Grid.py`. Here, outside the main codebase, diverse models have undergone scrutiny, with hyperparameter tuning conducted to enhance performance. These optimized parameters are subsequently inserted in the main code for execution. The decision to segregate the grid search from the main codebase primarily stems from considerations regarding code execution efficiency, deemed beyond the scope of the main codebase. Conceptually akin to prior operations, operations in this section are encapsulated and elucidated below.

Each file corresponds to a specific model and orchestrates a grid search. Data management responsibilities are delegated to `Evaluation.py` within the `DataPreparation` directory. This file encapsulates three pivotal functions: `load_data` (for data loading and dataframe definition), `normalize_data` (for normalization), and `dataset_preparation` (for train-test split and category definition). These functions constitute the initial phase of the code. Subsequently, following the definition of train and test objects for the complete dataset, the `eta` columns are systematically dropped from each dataset, mirroring actions undertaken in the main code.

The subsequent phase involves the implementation of grid search. Analogous to 'main.py' in the main codebase where models and parameters are specified, here is the breakdown:

- For BDT, parameters are defined as follows: 'n_estimators' (100, 125, 150), 'learning_rate' (0.15, 0.2, 0.25), 'max_depth' (2, 3, 4)
- For kNN, parameters are defined as: 'n_neighbors' (25, 50, 75), 'weights' ('uniform', 'distance'), 'metric' ('euclidean', 'manhattan')
- For RandomForest, parameters are defined as: 'n_estimators' (25, 50, 75), 'max_depth' (8, 10, 12), 'min_samples_split' (2, 3, 4)
- For NN, parameters are defined as: 'neurons' (32, 64, 128), 'drop_out' (0.2, 0.3, 0.4) 'learning_rate' (0.001, 0.01, 0.1)

Subsequently, a function "perform_grid_search" (located in `Grid.py`) is introduced, employing `GridSearchCV` from `sklearn.model_selection` for grid search, with `cv=3`, utilizing a 3-fold cross-validation approach during the search, and 'roc_auc' as the scoring metric to evaluate model performance.

Post-search, the best model (`best_model`), optimal parameters (`grid_search.best_params_`), and the highest achieved ROC AUC score (`grid_search.best_score_`) are returned for all three models. The final segment of the code pertains to result visualization, exclusively managed by the functions: `evaluate_model`, `evaluate_model_combined`, `print_plot` in `Evaluation.py`. Their role involves extracting `f1_score`, `accuracy`, and `precision` for the model working on the complete dataset, and combined models of the two categories (`evaluate_model` and `evaluate_model_combined` respectively), returning the `fpr` and `tpr` values for each case. Lastly, the `print_plot` function singularly addresses graphical visualization of the ROC for the obtained models.

The iperparameters that emerges from the application of the grid seearch are for the full dataset, category 1 and category 2, respectively:
- For BDT: 'n_estimators' (125, 100, 100), 'learning_rate' (0.15, 0.15, 0.15), 'max_depth' (3, 2, 3)
- For kNN, parameters are defined as: 'n_neighbors' (75, 75, 75), 'weights' ('uniform', 'uniform', 'distance'), 'metric' ('euclidean', 'manhattan','euclidean')
- For Random Forest, parameters are defined as: 'n_estimators' (75, 75, 75), 'max_depth' (8, 8, 8), 'min_samples_split' (2, 3, 3)
- For NN, parameters are defined as: 'neurons' (32, 64, 32), 'drop_out' (0.2, 0.3, 0.2) 'learning_rate' (0.01, 0.001, 0.01)