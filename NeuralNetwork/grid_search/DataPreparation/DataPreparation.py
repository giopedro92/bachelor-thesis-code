import os
import uproot
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(file1_path, treeS_name, treeB_name): # Defining the DataFrame signal and bkg
    # Uploading data
    file1 = uproot.open(file1_path)
    
    treeS = file1[treeS_name]
    treeB = file1[treeB_name]

    # Load data into DataFrame Pandas
    df_signal = treeS.arrays(library="pd")
    df_background = treeB.arrays(library="pd")

    # Select features and target separately for signal and background
    X_signal = df_signal[['var1', 'var2', 'var3', 'var4', 'eta']]
    X_background = df_background[['var1', 'var2', 'var3', 'var4', 'eta']]

    return X_signal, X_background


def normalize_data(X_signal, X_background): # Normalization function 
    X_signal_normalized = pd.DataFrame()
    X_background_normalized = pd.DataFrame()
    
    # Normalising data per signal by dividing by the maximum
    max_var1_signal = X_signal['var1'].max()
    max_var2_signal = X_signal['var2'].max()
    max_var3_signal = X_signal['var3'].max()
    max_var4_signal = X_signal['var4'].max()
    
    X_signal_normalized['var1'] = X_signal['var1'] / max_var1_signal
    X_signal_normalized['var2'] = X_signal['var2'] / max_var2_signal
    X_signal_normalized['var3'] = X_signal['var3'] / max_var3_signal
    X_signal_normalized['var4'] = X_signal['var4'] / max_var4_signal
    X_signal_normalized['eta'] = X_signal['eta']

    # Normalising data per background by dividing by the maximum
    max_var1_background = X_background['var1'].max()
    max_var2_background = X_background['var2'].max()
    max_var3_background = X_background['var3'].max()
    max_var4_background = X_background['var4'].max()

    X_background_normalized['var1'] = X_background['var1'] / max_var1_background
    X_background_normalized['var2'] = X_background['var2'] / max_var2_background
    X_background_normalized['var3'] = X_background['var3'] / max_var3_background
    X_background_normalized['var4'] = X_background['var4'] / max_var4_background
    X_background_normalized['eta']= X_background['eta']

    return X_signal_normalized, X_background_normalized
    

# Create dataset for training and test for full dataset, cat1&cat2
def dataset_preparation(X_signal_normalized, X_background_normalized, test_size = 0.2): 
    # Concatenates normalised DataFrames
    X_normalized = pd.concat([X_signal_normalized, X_background_normalized])
    # Define label 1 = signal, 0 = bkg
    y = np.concatenate([np.ones(len(X_signal_normalized)), np.zeros(len(X_background_normalized))])
    
    # Split normalised data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=test_size, random_state=42)
    
    # Divide the data into two groups according to the absolute value of 'age'
    X_train_cat1 = X_train[X_train['eta'].abs() > 1.3]
    X_train_cat2 = X_train[X_train['eta'].abs() <= 1.3]
    
    y_train_cat1 = y_train[X_train['eta'].abs() > 1.3]
    y_train_cat2 = y_train[X_train['eta'].abs() <= 1.3]
    
    X_test_cat1 = X_test[X_test['eta'].abs() > 1.3]
    X_test_cat2 = X_test[X_test['eta'].abs() <= 1.3]

    y_test_cat1 = y_test[X_test['eta'].abs() > 1.3]
    y_test_cat2 = y_test[X_test['eta'].abs() <= 1.3]
    
    
    # Dropping 'eta' column
    X_train = X_train.drop(columns=['eta'])
    X_test = X_test.drop(columns=['eta'])
    
    X_train_cat1 = X_train_cat1.drop(columns=['eta'])
    X_test_cat1 = X_test_cat1.drop(columns=['eta'])
    
    X_train_cat2 = X_train_cat2.drop(columns=['eta'])
    X_test_cat2 = X_test_cat2.drop(columns=['eta'])
    
    return X_train, X_test, y_train, y_test, X_train_cat1, X_train_cat2, y_train_cat1, y_train_cat2, X_test_cat1, X_test_cat2, y_test_cat1, y_test_cat2
