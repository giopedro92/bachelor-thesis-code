############################### Start import ###############################
print("--------------------START IMPORT--------------------")

import os
import sys
import numpy                   as np
import include.DataPreparation as dl  # Module for loading data
import include.Classifier      as clf # Module for defining classifiers
import include.MetricPrinter   as mp  # Module for printing metrics

############################### End import ###############################

############################### Start of code ###############################
print("--------------------START OF THE CODE--------------------")

"""
Checks whether the script is executed as a main program
and checks whether the correct number of arguments has been supplied
from the command line.

For example: one should verify that the execution of this code using a bash
file would be python3 main.py <name_model>.
If one wishes to execute this code independently, 
one should enter the following command into the terminal 
(it is not necessary to call other modules).
"""

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 main.py <model_type>")
        sys.exit(1)
    # Retrieve the model type from command-line arguments
    model_type = sys.argv[1]

    if not os.path.exists("evaluation_results"):
        os.makedirs("evaluation_results")

    # Dataset Signal and Background
    file1_path = 'root-trees/SB_simul.root'

    ############################### Preparing data ###############################
    print("--------------------PREPARING DATA--------------------")

    print("----------------------------------------DataPreparation--------------------")
    data_prep = dl.DataPreparation(file1_path)

    print("----------------------------------------load_data--------------------")
    data_prep.load_data()

    print("----------------------------------------prepare_data--------------------")
    data_prep.prepare_data()

    ############################### Model Definition ###############################
    print("--------------------MODEL DEFINITION--------------------")

    print("----------------------------------------SignalBackgroundClassifier--------------------")
    if model_type in ["BDT", "Neural_Network", "Random_Forest", "SVT", "kNN"]:
        classifier = clf.SignalBackgroundClassifier(
            model_type   = model_type,
            X_train      = data_prep.X_train,
            X_train_cat1 = data_prep.X_train_cat1,
            X_train_cat2 = data_prep.X_train_cat2,
        )
    else:
        print("The model in question is not supported.")
        sys.exit(1)

    ############################### Training & Evaluation ###############################
    print("--------------------TRAINING & EVALUATION--------------------")

    # Proviamo a estrarre le singole variabili dal tree e allenare la rete su ciascuna

    print("----------------------------------------train_classifier--------------------")
    classifier.train_classifier(
        data_prep.X_train,
        data_prep.y_train,
        # data_prep.X_train_cat1,
        # data_prep.y_train_cat1,
        # data_prep.X_train_cat2,
        # data_prep.y_train_cat2,
        data_prep.feature_names,
        model_type,
    )
    print("Training time ({}):".format(model_type), classifier.training_time)
    
    print("----------------------------------------evaluate_classifier--------------------")
    classifier.evaluate_classifier(data_prep.X_test,
                                   data_prep.y_test,
                                   # data_prep.X_test_cat1,
                                   # data_prep.y_test_cat1,
                                   # data_prep.X_test_cat2,
                                   # data_prep.y_test_cat2,
    )

############################### Print Results ###############################
print("--------------------PRINT RESULTS--------------------")

# Define the folder path
folder1_path = "evaluation_results"

# Define the full path of the output file
output_file = os.path.join(folder1_path, model_type + ".txt")

# It is necessary to create the output file required for the Metrics module
with open(output_file, "w") as f:
    f.write("accuracy:  {}\n".format(classifier.accuracy))  # Accuracy  result
    f.write("f1 score:  {}\n".format(classifier.f1))        # f1_score  result
    f.write("precision: {}\n".format(classifier.precision)) # precision result
    f.write("ROC AUC:   {}\n".format(classifier.roc_auc))     # roc auc   result
    f.write("fpr\ttpr\n")  # Printing of TPR and FPR data for ROC plot.
    for i in range(len(classifier.fpr)):
        f.write("{}\t{}\n".format(classifier.fpr[i], classifier.tpr[i]))

    ############################### Metrics display ###############################
    print("--------------------METRICS DISPLAY--------------------")

    # Print Roc and Confusion Matrix
    print("----------------------------------------PrintMetrics--------------------")
    metrics_printer = mp.PrintMetrics(classifier.fpr,
                                      classifier.tpr,
                                      # classifier.fpr_combined,
                                      # classifier.tpr_combined,
                                      classifier.accuracy,
                                      classifier.f1,
                                      classifier.precision,
                                      classifier.roc_auc,
                                      # classifier.accuracy_combined,
                                      # classifier.f1_combined,
                                      # classifier.precision_combined,
                                      data_prep.y_test,
                                      # classifier.y_test_combined,
                                      classifier.predictions,
                                      # classifier.predictions_combined,
                                     )
    
    print("----------------------------------------plot_roc_curve--------------------")    
    metrics_printer.plot_roc_curve()

    print("----------------------------------------print_metrics--------------------")
    metrics_printer.print_metrics()

############################### End ###############################
print("--------------------END--------------------")