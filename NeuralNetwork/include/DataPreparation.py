import uproot
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataPreparation:
    def __init__(self, file1_path, treeS_name="TreeS", treeB_name="TreeB"):
        '''
        Constructor to initialize the DataPreparation object.
        Parameters:
        - file1_path (str): Path to the ROOT file containing data.
        - treeS_name (str): Name of the signal tree. Default is "TreeS".
        - treeB_name (str): Name of the background tree. Default is "TreeB".
        This constructor sets the file path and tree names for the DataPreparation object.
        '''
        # Initialize the DataPreparation object with the ROOT file path and tree names
        self.file1_path = file1_path
        self.treeS_name = treeS_name
        self.treeB_name = treeB_name

    def load_data(self): # Loading data
        
        file1 = uproot.open(self.file1_path)
        
        # Extract signal and background trees
        treeS = file1[self.treeS_name]
        treeB = file1[self.treeB_name]

        # Load data from trees into Pandas DataFrames
        self.df_signal = treeS.arrays(library="pd")
        self.df_background = treeB.arrays(library="pd")
        
        
    def prepare_data(self): # This function is designed to prepare data for the training and evaluation of a classification model.
        # Select features
        X_signal = self.df_signal[['var1', 'var2', 'var3', 'var4','eta']]
        X_background = self.df_background[['var1', 'var2', 'var3', 'var4','eta']]
        self.feature_names=['var1', 'var2', 'var3', 'var4','eta']

        ####### Normalize signal data by dividing by the maximum value of each variable #######
        
        max_var1_signal = X_signal['var1'].max()
        max_var2_signal = X_signal['var2'].max()
        max_var3_signal = X_signal['var3'].max()
        max_var4_signal = X_signal['var4'].max()

        X_signal_normalized = pd.DataFrame()
        X_background_normalized = pd.DataFrame()

        X_signal_normalized['var1'] = X_signal['var1'] / max_var1_signal
        X_signal_normalized['var2'] = X_signal['var2'] / max_var2_signal
        X_signal_normalized['var3'] = X_signal['var3'] / max_var3_signal
        X_signal_normalized['var4'] = X_signal['var4'] / max_var4_signal
        X_signal_normalized['eta'] = X_signal['eta'] # Noted that the data has not been normalised for the purpose of categorisation.

        # Normalizzazione dei dati per sfondo dividendo per il massimo
        max_var1_background = X_background['var1'].max()
        max_var2_background = X_background['var2'].max()
        max_var3_background = X_background['var3'].max()
        max_var4_background = X_background['var4'].max()

        X_background_normalized['var1'] = X_background['var1'] / max_var1_background
        X_background_normalized['var2'] = X_background['var2'] / max_var2_background
        X_background_normalized['var3'] = X_background['var3'] / max_var3_background
        X_background_normalized['var4'] = X_background['var4'] / max_var4_background
        X_background_normalized['eta']= X_background['eta'] # Noted that the data has not been normalised for the purpose of categorisation.

        ####### End Normalisation #######
        
        # Concatenate normalized DataFrames
        X = pd.concat([X_signal_normalized, X_background_normalized])
        
        # Add a 'target' column to distinguish signal (1) from background (0)
        y = np.concatenate([np.ones(len(X_signal_normalized)), np.zeros(len(X_background_normalized))])

        # Split data into training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)


        ####### Definition of category #######
        
        # Separate data into two groups based on the absolute value of "eta" => categorisation
        self.X_train_cat1 = self.X_train[self.X_train['eta'].abs() > 1.3]
        self.X_train_cat2 = self.X_train[self.X_train['eta'].abs() <= 1.3]

        self.y_train_cat1 = self.y_train[self.X_train['eta'].abs() > 1.3]
        self.y_train_cat2 = self.y_train[self.X_train['eta'].abs() <= 1.3]

        self.X_test_cat1 = self.X_test[self.X_test['eta'].abs() > 1.3]
        self.X_test_cat2 = self.X_test[self.X_test['eta'].abs() <= 1.3]
        
        self.y_test_cat1 = self.y_test[self.X_test['eta'].abs() > 1.3]
        self.y_test_cat2 = self.y_test[self.X_test['eta'].abs() <= 1.3]
        
        # Dropping 'eta' column
        self.X_train = self.X_train.drop(columns=['eta'])
        self.X_test = self.X_test.drop(columns=['eta'])
        self.X_train_cat1 = self.X_train_cat1.drop(columns=['eta'])
        self.X_test_cat1 = self.X_test_cat1.drop(columns=['eta'])
        self.X_train_cat2 = self.X_train_cat2.drop(columns=['eta'])
        self.X_test_cat2 = self.X_test_cat2.drop(columns=['eta'])
