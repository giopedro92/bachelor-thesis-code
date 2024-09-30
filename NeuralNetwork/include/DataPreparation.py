import uproot
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


class DataPreparation:

    ############################### INIT ###############################

    def __init__(self, file1_path, treeS_name="TreeS", treeB_name="TreeB"):
        """
        CONSTRUCTOR to initialize the DataPreparation object.
        Parameters:
        - file1_path (str): Path to the ROOT file containing data.
        - treeS_name (str): Name of the signal tree. Default is "TreeS".
        - treeB_name (str): Name of the background tree. Default is "TreeB".
        This constructor sets the file path and tree names for the DataPreparation object.
        """

        # Initialize the DataPreparation OBJECT with the ROOT file path and tree names
        self.file1_path = file1_path
        self.treeS_name = treeS_name
        self.treeB_name = treeB_name

    ############################### LOAD DATA ###############################

    def load_data(self):  # Loading data
        print("------------------------------------------------------------uproot.open--------------------")
        file1 = uproot.open(self.file1_path)

        print("------------------------------------------------------------extract--------------------")
        # Extract signal and background trees
        treeS = file1[self.treeS_name]
        treeB = file1[self.treeB_name]

        # Print what imported
        print(treeS)
        print(treeB)
        print("\n")

        treeS.show()
        print("\n")
        treeB.show()
        print("\n")

        print("------------------------------------------------------------.arrays--------------------")
        # Load data from Trees into Pandas DataFrames (df)
        self.df_signal = treeS.arrays(library="pd")
        self.df_background = treeB.arrays(library="pd")

    ############################### PREPARE DATA ###############################

    def prepare_data(self):  # This function is designed to prepare data for the training and evaluation of a classification model.
        # Select features
        print("------------------------------------------------------------X_signal--------------------")
        X_signal = self.df_signal[["massK0S", "tImpParBach", "tImpParV0", "CtK0S", "cosPAK0S", "nSigmapr", "dcaV0"]]

        print("------------------------------------------------------------X_background--------------------")
        X_background = self.df_background[["massK0S", "tImpParBach", "tImpParV0", "CtK0S", "cosPAK0S", "nSigmapr", "dcaV0"]]

        print("------------------------------------------------------------X_feature_names--------------------")
        self.feature_names = ["massK0S", "tImpParBach", "tImpParV0", "CtK0S", "cosPAK0S", "nSigmapr", "dcaV0"]

        # Normalize SIGNAL data by dividing by the maximum value of each variable
        max_massK0S_signal = X_signal["massK0S"].max()
        max_tImpParBach_signal = X_signal["tImpParBach"].max()
        max_tImpParV0_signal = X_signal["tImpParV0"].max()
        max_CtK0S_signal = X_signal["CtK0S"].max()
        max_cosPAK0S_signal = X_signal["cosPAK0S"].max()
        max_nSigmapr_signal = X_signal["nSigmapr"].max()
        max_dcaV0_signal = X_signal["dcaV0"].max()

        X_signal_normalized = pd.DataFrame()

        X_signal_normalized["massK0S"] = X_signal["massK0S"] / max_massK0S_signal
        X_signal_normalized["tImpParBach"] = (X_signal["tImpParBach"] / max_tImpParBach_signal)
        X_signal_normalized["tImpParV0"] = X_signal["tImpParV0"] / max_tImpParV0_signal
        X_signal_normalized["CtK0S"] = X_signal["CtK0S"] / max_CtK0S_signal
        X_signal_normalized["cosPAK0S"] = X_signal["cosPAK0S"] / max_cosPAK0S_signal
        X_signal_normalized["nSigmapr"] = X_signal["nSigmapr"] / max_nSigmapr_signal
        X_signal_normalized["dcaV0"] = X_signal["dcaV0"] / max_dcaV0_signal

        # Normalize BACKGROUND data by dividing by the maximum value of each variable
        max_massK0S_background = X_background["massK0S"].max()
        max_tImpParBach_background = X_background["tImpParBach"].max()
        max_tImpParV0_background = X_background["tImpParV0"].max()
        max_CtK0S_background = X_background["CtK0S"].max()
        max_cosPAK0S_background = X_background["cosPAK0S"].max()
        max_nSigmapr_background = X_background["nSigmapr"].max()
        max_dcaV0_background = X_background["dcaV0"].max()

        X_background_normalized = pd.DataFrame()

        X_background_normalized["massK0S"] = (X_background["massK0S"] / max_massK0S_background)
        X_background_normalized["tImpParBach"] = (X_background["tImpParBach"] / max_tImpParBach_background)
        X_background_normalized["tImpParV0"] = (X_background["tImpParV0"] / max_tImpParV0_background)
        X_background_normalized["CtK0S"] = X_background["CtK0S"] / max_CtK0S_background
        X_background_normalized["cosPAK0S"] = (X_background["cosPAK0S"] / max_cosPAK0S_background)
        X_background_normalized["nSigmapr"] = (X_background["nSigmapr"] / max_nSigmapr_background)
        X_background_normalized["dcaV0"] = X_background["dcaV0"] / max_dcaV0_background

        ############################### End Normalisation ###############################



        # Concatenate normalized DataFrames
        X = pd.concat([X_signal_normalized, X_background_normalized])

        # Add a 'target' column to distinguish signal (1) from background (0)
        y = np.concatenate(
            [np.ones(len(X_signal_normalized)), np.zeros(len(X_background_normalized))]
        )

        # Split data into training and test sets
        self.X_train,
        self.X_test,
        self.y_train,
        self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        ############################### Definition of category ###############################
"""        
        # Separate data into two groups based on the absolute value of "eta" => categorisation
        self.X_train_cat1 = self.X_train[self.X_train['eta'].abs() > 1.3]
        self.X_train_cat2 = self.X_train[self.X_train['eta'].abs() <= 1.3]

        self.y_train_cat1 = self.y_train[self.X_train['eta'].abs() > 1.3]
        self.y_train_cat2 = self.y_train[self.X_train['eta'].abs() <= 1.3]

        self.X_test_cat1  = self.X_test[self.X_test['eta'].abs() > 1.3]
        self.X_test_cat2  = self.X_test[self.X_test['eta'].abs() <= 1.3]
        
        self.y_test_cat1  = self.y_test[self.X_test['eta'].abs() > 1.3]
        self.y_test_cat2  = self.y_test[self.X_test['eta'].abs() <= 1.3]
        
        # Dropping 'eta' column
        self.X_train      = self.X_train.drop(columns=['eta'])
        self.X_test       = self.X_test.drop(columns=['eta'])
        self.X_train_cat1 = self.X_train_cat1.drop(columns=['eta'])
        self.X_test_cat1  = self.X_test_cat1.drop(columns=['eta'])
        self.X_train_cat2 = self.X_train_cat2.drop(columns=['eta'])
        self.X_test_cat2  = self.X_test_cat2.drop(columns=['eta'])
"""
