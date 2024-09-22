############################### Start import ###############################

import os
import sys
import numpy                   as np
import include.DataPreparation as dl  # Module for loading data
import include.Classifier      as clf # Module for defining classifiers
import include.MetricPrinter   as mp  # Module for printing metrics

############################### End import ###############################

############################### Start of code ###############################
'''
Checks whether the script is executed as a main program
and checks whether the correct number of arguments has been supplied
from the command line.

For example: one should verify that the execution of this code using a bash
file would be Python3 main.py <name_model>.
If one wishes to execute this code independently, 
one should enter the following command into the terminal 
(it is not necessary to call other modules).
'''
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 main.py <model_type>")
        sys.exit(1)
    # Retrieve the model type from command-line arguments
    model_type = sys.argv[1]
    
    if not os.path.exists("evaluation_results"):
        os.makedirs("evaluation_results")
        
    # Dataset
    ## Signal
    fileS_path = '../root-trees/signalNew_1_2.root'
    ## Background
    fileB_path = '../root-trees/background_1_2.root'

    ############################### Preparing data ###############################

    data_prep = dl.DataPreparation(fileS_path)
    data_prep.load_data()
    data_prep.prepare_data()

    data_prep = dl.DataPreparation(fileB_path)
    data_prep.load_data()
    data_prep.prepare_data()

    ############################### Model Definition ###############################
    
    if model_type in ['BDT', 'Neural_Network', 'Random_Forest', 'SVT', 'kNN']:
        classifier = clf.SignalBackgroundClassifier(model_type   = model_type,
                                                    X_train      = data_prep.X_train,
                                                    X_train_cat1 = data_prep.X_train_cat1,
                                                    X_train_cat2 = data_prep.X_train_cat2)
    
    else:
        print("The model in question is not supported.")
        sys.exit(1)
    
    ############################### Training & Evaluation ###############################
    
    classifier.train_classifier(data_prep.X_train,
                                data_prep.y_train,
                                data_prep.X_train_cat1,
                                data_prep.y_train_cat1,
                                data_prep.X_train_cat2,
                                data_prep.y_train_cat2,
                                data_prep.feature_names,
                                model_type)
    print("Training time ({}):".format(model_type),
          classifier.training_time)
    classifier.evaluate_classifier(data_prep.X_test,
                                   data_prep.y_test,
                                   data_prep.X_test_cat1,
                                   data_prep.y_test_cat1,
                                   data_prep.X_test_cat2,
                                   data_prep.y_test_cat2)
    
    