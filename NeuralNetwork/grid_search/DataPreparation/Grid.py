from sklearn.model_selection import GridSearchCV

def perform_grid_search(clf, param_grid, X_train, y_train):
    """
    Perform grid search to find the best hyperparameters for a classifier.

    Args:
        clf: Classifier object to perform grid search on.
        param_grid: Dictionary or list of dictionaries specifying the parameters to test during grid search.
        X_train: Training feature set.
        y_train: Training label set.

    Returns:
        best_model: Best model found during grid search.
        best_params: Best parameters found during grid search.
        best_score: Best ROC AUC score achieved by the best model.
    """
    # Initialize grid search
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, scoring='roc_auc', n_jobs=-1)

    # Perform grid search on training set
    grid_search.fit(X_train, y_train)

    # Get the best model found by grid search
    best_model = grid_search.best_estimator_

    return best_model, grid_search.best_params_, grid_search.best_score_
