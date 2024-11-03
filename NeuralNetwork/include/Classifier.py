import time
import numpy               as np
import tensorflow          as tf
import matplotlib.pyplot   as plt

from sklearn.svm           import SVC
from sklearn.ensemble      import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors     import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import (accuracy_score,
                                   roc_curve,
                                   roc_auc_score,
                                   f1_score,
                                   precision_score,
                                   classification_report)
from tensorflow            import keras
from tensorflow.keras      import layers


############################### CLASS for model definition, training and evaluation ###############################

class SignalBackgroundClassifier:

    ############################### DATA PREPARATION ###############################

    def __init__(
        self,
        model_type   = "Neural_Network",
        X_train      = None,
        X_train_cat1 = None,
        X_train_cat2 = None,
    ):
        """
        Initialize the SignalBackgroundClassifier object.

        Parameters:
        - model_type (str): Type of model to be used. Default is 'Neural_Network'.
        - X_train (array-like): Training data for the main model.
        - X_train_cat1 (array-like): Training data for the first category.
        - X_train_cat2 (array-like): Training data for the second category.

        Raises:
        - ValueError: If an invalid model_type is provided.
        """

        ############################### Model definition ###############################

        # Initialize the classifier based on the specified model type
        if model_type == "BDT":
            self.clf = GradientBoostingClassifier(
                n_estimators=125, learning_rate=0.15, max_depth=3, random_state=42
            )
            self.clf_cat1 = GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.15, max_depth=2, random_state=42
            )
            self.clf_cat2 = GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.15, max_depth=3, random_state=42
            )
        
        elif model_type == "SVT":
            self.clf = SVC(kernel="linear", probability=True)
            self.clf_cat1 = SVC(kernel="linear", probability=True)
            self.clf_cat2 = SVC(kernel="linear", probability=True)
        
        elif model_type == "Neural_Network":
            self.clf = NeuralNetwork(X_train,
                                     32,
                                     0.2,
                                     0.01)
            # self.clf_cat1 = NeuralNetwork(X_train_cat1, 64, 0.3, 0.001)
            # self.clf_cat2 = NeuralNetwork(X_train_cat2, 32, 0.2, 0.01)
        
        elif model_type == "Random_Forest":
            self.clf = RandomForestClassifier(
                n_estimators=75, max_depth=8, min_samples_split=2, random_state=42
            )
            self.clf_cat1 = RandomForestClassifier(
                n_estimators=75, max_depth=8, min_samples_split=3, random_state=42
            )
            self.clf_cat2 = RandomForestClassifier(
                n_estimators=75, max_depth=8, min_samples_split=2, random_state=42
            )
        
        elif model_type == "kNN":
            self.clf = KNeighborsClassifier(
                n_neighbors=75, metric="euclidean", weights="uniform"
            )
            self.clf_cat1 = KNeighborsClassifier(
                n_neighbors=75, metric="manhattan", weights="uniform"
            )
            self.clf_cat2 = KNeighborsClassifier(
                n_neighbors=75, metric="euclidean", weights="distance"
            )
        
        else:
            raise ValueError("Invalid model type.")

    ############################### Training ###############################

    def train_classifier(self,
                         X_train,
                         y_train,
                        #  X_train_cat1,
                        #  y_train_cat1,
                        #  X_train_cat2,
                        #  y_train_cat2,
                         feature_names,
                         model_type,
                        ):  # Training function

        # Train the classifier NN, noting the call to the class NeuralNetwork
        # and the function contained therein, namely build_model
        if isinstance(self.clf, NeuralNetwork):
            # self.clf_cat1.build_model(X_train_cat1, 64, 0.3, 0.01)
            # self.clf_cat2.build_model(X_train_cat2, 64, 0.3, 0.01)
            self.clf.build_model(X_train,
                                 64,
                                 0.3,
                                 0.01)
# 64
# 0.3
# 0.01

            # Start measuring training time
            start_time = time.time()
            self.clf.train_classifier(X_train,
                                      y_train)
            # Calculate training time
            self.training_time = time.time() - start_time

            # self.clf_cat1.train_classifier(X_train_cat1, y_train_cat1)
            # self.clf_cat2.train_classifier(X_train_cat2, y_train_cat2) ########### ERRORE QUI

        else:  # Train the other classifier
            # Start measuring training time
            start_time = time.time()
            self.clf.fit(X_train,
                         y_train)
            # Calculate training time
            self.training_time = time.time() - start_time

            # self.clf_cat1.fit(X_train_cat1, y_train_cat1)
            # self.clf_cat2.fit(X_train_cat2, y_train_cat2)

        # Create an instance of the Additional_evaluation class
        # and then call the methods associated with it.
        additional_evaluation = Additional_evaluation(self.clf,
                                                      feature_names,
                                                      model_type)
        additional_evaluation.plot_feature_importance(self.clf,
                                                      X_train,
                                                      y_train)

    ############################### Evaluating ###############################

    def evaluate_classifier(self,
                            X_test,
                            y_test,
                            # X_test_cat1,
                            # y_test_cat1,
                            # X_test_cat2,
                            # y_test_cat2,
                           ):
        
        # Formulate predictions for NN
        print("------------------------------------------------------------probability--------------------")
        if isinstance(self.clf, NeuralNetwork):
            probability = self.clf.model.predict(X_test)
            # probability_cat1 = self.clf_cat1.model.predict(X_test_cat1)
            # probability_cat2 = self.clf_cat2.model.predict(X_test_cat2)

        else:  # Formulate predictions for the performance of other classifiers.
            probability = self.clf.predict_proba(X_test)[:, 1]
            # probability_cat1 = self.clf_cat1.predict_proba(X_test_cat1)[:, 1]
            # probability_cat2 = self.clf_cat2.predict_proba(X_test_cat2)[:, 1]

        # Calculate predicted probabilities for the positive class for the two cases
        print("------------------------------------------------------------predictions--------------------")
        if isinstance(self.clf, NeuralNetwork):
            self.predictions = (probability > 0.5).astype(int)
            # predictions_cat1 = (probability_cat1 > 0.5).astype(int)
            # predictions_cat2 = (probability_cat2 > 0.5).astype(int)
        else:
            self.predictions = self.clf.predict(X_test)
            # predictions_cat1 = self.clf_cat1.predict(X_test_cat1)
            # predictions_cat2 = self.clf_cat2.predict(X_test_cat2)

        # print("------------------------------------------------------------predictions_categories--------------------")
        # # Combine predictions and predicted probabilities for both categories
        # self.predictions_combined = np.concatenate(
        #     [predictions_cat1, predictions_cat2]
        # )  # Needed for Confusion Matrix
        # probability_combined = np.concatenate([probability_cat1, probability_cat2])
        # self.y_test_combined = np.concatenate(
        #     [y_test_cat1, y_test_cat2]
        # )  # Needed for Confusion Matrix

        print("------------------------------------------------------------calculate_roc_curve--------------------")
        # Calculate ROC curve
        self.fpr, self.tpr, self.thresholds = roc_curve(y_test,     probability)
        self.roc_auc                        = roc_auc_score(y_test, probability)

        # print("------------------------------------------------------------calculate_roc_curve_categories--------------------")
        # # ROC curve is to be calculated by combining two categories.
        # self.fpr_combined, self.tpr_combined, _ = roc_curve(
        #     self.y_test_combined, probability_combined
        # )
        # self.roc_auc_combined = roc_auc_score(
        #     self.y_test_combined, probability_combined
        # )

        print("------------------------------------------------------------calculate_metrics--------------------")
        # Calculate model accuracy, f1_score, precision
        # Full dataset
        self.accuracy  = accuracy_score(y_test,  self.predictions)
        self.f1        = f1_score(y_test,        self.predictions)
        self.precision = precision_score(y_test, self.predictions)

        # print("------------------------------------------------------------calculate_metrics_categories--------------------")
        # # Categorisation
        # self.accuracy_combined = accuracy_score(
        #     self.y_test_combined, self.predictions_combined
        # )
        # self.f1_combined = f1_score(self.y_test_combined, self.predictions_combined)
        # self.precision_combined = precision_score(
        #     self.y_test_combined, self.predictions_combined
        # )

        print("------------------------------------------------------------print--------------------")
        print("Full dataset :")
        print(classification_report(y_test,
                                    self.predictions))
        # print("Categorisation :")
        # print(classification_report(self.y_test_combined, self.predictions_combined))


############################### CLASS NN ###############################

class NeuralNetwork:
    def __init__(self,
                 X_train,
                 neurons,
                 drop_out,
                 learning_rate):
        print("----------------------------------------NeuralNetwork--------------------")
        """
        Constructor for initializing the NeuralNetwork class.
        Parameters:
        - X_train (array-like): Training data for the neural network.
        This constructor creates and initializes the neural network model using the provided training data.
        """

        self.model = self.build_model(X_train,
                                      neurons,
                                      drop_out,
                                      learning_rate)

    ############################### Model definition ###############################
    # COSTRUZIONE DEL MODELLO
    def build_model(self,
                    X_train,
                    neurons,
                    drop_out,
                    learning_rate):
        print("----------------------------------------NeuralNetworkbuild_model--------------------")
        model = keras.Sequential(
            [
                layers.Input(
                    shape = (X_train.shape[1],)
                ),  # layer di input, shape è dimensione dei dati
                layers.Dense(
                    neurons,
                    activation = "relu"
                ),  # layer collegati a tutti i neuroni
                layers.Dropout(
                    drop_out
                ),  # spegne un tot di neuroni per non influenzare troppo la rete
                layers.Dense(
                    neurons,
                    activation = "relu"),
                layers.Dropout(
                    drop_out),
                layers.Dense(1,
                             activation = "sigmoid")])

        optimizer = keras.optimizers.Adam(
            learning_rate = learning_rate
        )  # usa optimizer Adam

        model.compile(
            optimizer = optimizer,
            loss      = "binary_crossentropy",
            metrics   = ["accuracy"],
        )
        # loss function più adeguata per il problema
        return model

    ############################### Training of the model ###############################

    def train_classifier(self,
                         X_train,
                         y_train):
        print("----------------------------------------NeuralNetworktrain_classifier--------------------")
        # Start measuring time
        start_time = time.time()

        # funzione fondamentale con cui viene allenato il modello model.fit
        self.history = self.model.fit(X_train,
                                 y_train,
                                 epochs           = 10,
                                 batch_size       = 32,
                                 validation_split = 0.2)
        # DIVIDE TRAIN TRA TRAIN E VALIDATION validation_split=0.2

        # Calculates the elapsed time for training
        self.training_time = time.time() - start_time

############################### CLASS Additional tools ###############################

class Additional_evaluation:
    def __init__(self,
                 clf,
                 feature_names,
                 model_type):
        """
        Constructor to initialize Additional_evaluation object with classifier, feature names, and model type.
        Parameters:
        - clf: Classifier object.
        - feature_names (list): List of feature names.
        - model_type (str): Type of the classifier model.
        """
        self.clf = clf
        self.feature_names = feature_names
        self.model_type = model_type

    # Plot feature importance for different classifier types
    def plot_feature_importance(self,
                                clf,
                                X_test,
                                y_test):
        """
        - barh: Draws a horizontal bar graph of the importance features, with the features on the y-axis and the importance on the bars.
        - Label the bars with the feature names
        - Label the x-axis as 'Importance'.
        """
        
        if self.model_type in ["BDT", "Random_Forest"]:
            if hasattr(self.clf, "feature_importances_"):
                # Calculate feature importance for BDT and Random Forest classifiers
                feature_importance = self.clf.feature_importances_
                sorted_idx = np.argsort(feature_importance)
                plt.figure(figsize=(10, 6))
                plt.barh(
                    range(len(sorted_idx)),
                    feature_importance[sorted_idx],
                    align="center",
                )
                plt.yticks(
                    range(len(sorted_idx)), [self.feature_names[i] for i in sorted_idx]
                )
                plt.xlabel("Importance")
                plt.title("Feature Importance")
                plt.show()
            else:
                print(
                    "Plotting feature importance is not supported for this type of classifier."
                )
        
        elif self.model_type == "SVT":
            if isinstance(self.clf, SVC) and self.clf.kernel == "linear":
                # Calculate feature importance for SVM with linear kernel
                weights = np.abs(self.clf.coef_).mean(axis=0)
                total_weight = np.sum(weights)
                weights_normalized = weights / total_weight
                sorted_indices = np.argsort(weights_normalized)[::-1]
                sorted_features = [self.feature_names[i] for i in sorted_indices]
                sorted_weights = weights_normalized[sorted_indices]

                plt.figure(figsize=(10, 6))
                plt.barh(sorted_features, sorted_weights)
                plt.xlabel("Feature Importance")
                plt.ylabel("Features")
                plt.title("Feature Importance for Support Vector Machine")
                plt.show()
            else:
                print(
                    "Plotting feature importance is not supported for this type of classifier."
                )
        
        elif self.model_type == "Neural_Network":
            if isinstance(self.clf, NeuralNetwork):
                # Calculate feature importance for Neural Network model
                weights         = np.abs(self.clf.model.get_weights()[0]).mean(axis=1)
                sorted_indices  = np.argsort(weights)[::-1]
                sorted_features = [self.feature_names[i] for i in sorted_indices]
                sorted_weights  = weights[sorted_indices]

                plt.figure(figsize=(10, 6))
                plt.barh(sorted_features,
                         sorted_weights)
                plt.xlabel("Feature Importance")
                plt.ylabel("Features")
                plt.title("Feature Importance for Neural Network")
                # plt.show()
                plt.savefig("evaluation_results/feature_importance_Neural_Network.svg")

            else:
                print(
                    "Plotting feature importance is not supported for this type of classifier."
                )
        
        elif self.model_type == "kNN":
            if isinstance(self.clf, KNeighborsClassifier):
                # Calculate feature importance for kNN classifier
                feature_indices = range(X_test.shape[1])
                _, indices = self.clf.kneighbors(X_test, n_neighbors=5)

                feature_counts = {}
                for neighbors in indices:
                    for neighbor in neighbors:
                        for feature_index in feature_indices:
                            feature_name = feature_index
                            if feature_name not in feature_counts:
                                feature_counts[feature_name] = 1
                            else:
                                feature_counts[feature_name] += 1

                total_counts = sum(feature_counts.values())
                feature_importance = {
                    feature_name: count / total_counts
                    for feature_name, count in feature_counts.items()
                }

                plt.figure(figsize=(10, 6))
                sorted_features = sorted(
                    feature_importance.items(), key=lambda x: x[1], reverse=True
                )
                features, importance = zip(*sorted_features)
                plt.barh(features, importance)
                plt.xlabel("Feature Importance")
                plt.ylabel("Features")
                plt.title("Feature Importance for kNN")
                plt.show()
            else:
                print(
                    "Feature importance calculation is not supported for this type of classifier."
                )
        
        else:
            print(
                "Plotting feature importance is not supported for {}.".format(
                    self.model_type
                )
            )