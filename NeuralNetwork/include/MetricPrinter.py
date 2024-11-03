import matplotlib.pyplot as plt
import seaborn           as sns

from sklearn.metrics     import (
    accuracy_score,
    roc_curve,
    roc_auc_score,
    f1_score,
    precision_score,
    confusion_matrix,
)


############################### CLASS to print metrics ###############################

class PrintMetrics:
    def __init__(
        self,
        fpr,
        tpr,
        # fpr_combined,
        # tpr_combined,
        accuracy,
        f1,
        precision,
        roc_auc,
        # accuracy_combined,
        # f1_combined,
        # precision_combined,
        y_test,
        # y_test_combined,
        predictions,
        # predictions_combined,
    ):
        """
        Constructor to initialize PrintMetrics object with metrics values and visualize confusion matrix.
        Parameters:
        - fpr (array-like):                  False Positive Rate values.
        - tpr (array-like):                  True Positive Rate values.
        - fpr_combined (array-like):         for combined categories.
        - tpr_combined (array-like):         for combined categories.
        - accuracy (float):                  Accuracy score of the classifier.
        - accuracy_combined (float):         for combined categories.
        - f1 (float):                        F1 score of the classifier.
        - f1_combined (float):               for combined categories.
        - precision (float):                 Precision score of the classifier.
        - precision_combined (float):        for combined categories.
        - y_test (array-like):               True labels for the test data.
        - y-test (array-like):               for combined categories.
        - predictions (array-like):          Predicted labels for the test data
        - predictions_combined (array-like): for combined categories.
        - roc_auc:                           area under the curve for roc curve
        """

        self.fpr         = fpr
        self.tpr         = tpr
        # self.fpr_combined = fpr_combined
        # self.tpr_combined = tpr_combined
        self.accuracy    = accuracy
        self.f1          = f1
        self.precision   = precision
        self.roc_auc     = roc_auc
        # self.accuracy_combined = accuracy_combined
        # self.f1_combined = f1_combined
        # self.precision_combined = precision_combined
        self.y_test      = y_test
        # self.y_test_combined = y_test_combined
        self.predictions = predictions
        # self.predictions_combined = predictions_combined

    def plot_roc_curve(self):
        # Calculate background rejection (1 - FPR)
        print("------------------------------------------------------------background_rejection--------------------")        
        
        background_rejection = 1 - self.fpr
        # background_rejection_combined = 1 - self.fpr_combined


        # Plot signal efficiency vs background rejection
        print("------------------------------------------------------------roc_curve--------------------")

        plt.figure(figsize=(8, 6))
        plt.plot(
            self.tpr,
            background_rejection,
            color = "blue",
            lw    = 2,
            label = "Full dataset"
        )
        # plt.plot(
        #     self.tpr_combined,
        #     background_rejection_combined,
        #     color="red",
        #     lw=2,
        #     label="Combined dataset",
        # )
        plt.xlabel("Signal Efficiency (True Positive Rate)")
        plt.ylabel("Background Rejection")
        plt.title("Signal Efficiency vs Background Rejection")
        plt.legend(loc="lower right")
        plt.grid(True)
        # plt.savefig(f"plot_results/{available_options[choice]}_{event_number}_{width}x{height}.png") # mettere che salva i plot
        plt.savefig("evaluation_results/roc_curve.svg")
        #plt.show()

    def print_metrics(self):
        # Print metrics
        print("------------------------------------------------------------print_metrics--------------------")        
        
        print("Accuracy: ", self.accuracy)
        print("F1 Score: ", self.f1)
        print("Precision:", self.precision)


        # print("Accuracy Combined: ", self.accuracy_combined)
        # print("F1 Score Combined: ", self.f1_combined)
        # print("Precision Combined:", self.precision_combined)


        print("ROC AUC:  ", self.roc_auc)


        # Confusion Matrix
        print("------------------------------------------------------------confusion_matrix--------------------")        
        cm = confusion_matrix(self.y_test,
                              self.predictions)

        # Visualize Confusion Matrix
        plt.figure(figsize = (8, 6))
        sns.heatmap(cm,
                    annot  = True,
                    fmt    = "d",
                    cmap   = "Blues")
        plt.title("Confusion Matrix (Full dataset)")
        plt.xlabel("Predicted values")
        plt.ylabel("True values")
        # plt.show()
        plt.savefig("evaluation_results/confusion_matrix.svg")


        # Confusion Matrix Combined
        #print("------------------------------------------------------------confusion_matrix_combined--------------------")        
        # cm_combined = confusion_matrix(self.y_test_combined, self.predictions_combined)

        # Visualize Confusion Matrix Combined
        # plt.figure(figsize=(8, 6))
        # sns.heatmap(cm_combined, annot=True, fmt="d", cmap="Blues")
        # plt.title("Confusion Matrix (Categorisation)")
        # plt.xlabel("Predicted values")
        # plt.ylabel("True values")
        # # plt.show()
        # plt.savefig("evaluation_results/confusion_matrix_combined.svg")