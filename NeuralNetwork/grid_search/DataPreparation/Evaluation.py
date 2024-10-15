import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, f1_score, precision_score

def evaluate_model(best_model, X_test, y_test): # Evaluation for full dataset
    
    # Make predictions
    predictions = best_model.predict(X_test)
    
    # Calculate the predicted probabilities for the positive class
    probabilities = best_model.predict_proba(X_test)[:, 1]
    
    # Calculates the accuracy, f1, precision of the model
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    
    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    print("Precision:", precision)


    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, probabilities)
    
    # Calculate AUC
    roc_auc = roc_auc_score(y_test, probabilities)

    # Calculate background rejection (1 - FPR)
    background_rejection = 1 - fpr
    
    return tpr, background_rejection



def evaluate_model_combined(best_model_cat1,best_model_cat2, X_test_cat1, y_test_cat1, X_test_cat2, y_test_cat2): #Evaluation for cat1 & cat2
    # Make predictions for both categories
    predictions_cat1 = best_model_cat1.predict(X_test_cat1)
    predictions_cat2 = best_model_cat2.predict(X_test_cat2)

    # Calculate the predicted probabilities for the positive class for both categories
    y_probs_cat1 = best_model_cat1.predict_proba(X_test_cat1)[:, 1]
    y_probs_cat2 = best_model_cat2.predict_proba(X_test_cat2)[:, 1]
    
    # Combination: 
    predictions_combined = np.concatenate([predictions_cat1, predictions_cat2])
    y_probs_combined = np.concatenate([y_probs_cat1, y_probs_cat2])
    y_test_combined = np.concatenate([y_test_cat1, y_test_cat2])
    
    # Calculates the accuracy, f1, precision of the model
    accuracy = accuracy_score(y_test_combined, predictions_combined)
    f1 = f1_score(y_test_combined, predictions_combined)
    precision = precision_score(y_test_combined, predictions_combined)
    
    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    print("Precision:", precision)
    

    # Calculate ROC curve for combination of categories
    fpr_combined, tpr_combined, thresholds_combined = roc_curve(y_test_combined, y_probs_combined)

    # Calculate AUC curve for combination of categories
    roc_auc_combined = roc_auc_score(y_test_combined, y_probs_combined)

    # Calculate background rejection (1 - FPR)
    background_rejection_combined = 1 - fpr_combined

    
    return tpr_combined, background_rejection_combined



def print_plot(tpr_combined, background_rejection_combined, tpr, background_rejection): # Plot comparative ROC
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(tpr, background_rejection, color='blue', lw=2, label='Full Dataset - Signal Efficiency vs Background Rejection')
    plt.plot(tpr_combined, background_rejection_combined, color='red', lw=2, label='Combined Dataset - Signal Efficiency vs Background Rejection')

    plt.xlabel('Signal Efficiency')
    plt.ylabel('Background Rejection')
    plt.title('Signal Efficiency vs Background Rejection')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
