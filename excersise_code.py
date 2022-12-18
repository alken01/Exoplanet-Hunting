import sys

sys.path.append('..')
import numpy as np
from scipy.io import loadmat
from os.path import join
from matplotlib import pyplot

# in this file we will add the code from the exercise
import numpy as np

# Code from: EX1. Optional Exercises: 3.1 Feature Normalization
def  featureNormalize(X):
    """
    Normalizes the features in X. returns a normalized version of X where
    the mean value of  xeach feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when working with
    learning algorithms.
    
    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n).
    
    Returns
    -------
    X_norm : array_like
        The normalized dataset of shape (m x n).
    
    Instructions
    ------------
    First, for each feature dimension, compute the mean of the feature
    and subtract it from the dataset, storing the mean value in mu. 
    Next, compute the  standard deviation of each feature and divide
    each feature by it's standard deviation, storing the standard deviation 
    in sigma. 
    
    Note that X is a matrix where each column is a feature and each row is
    an example. You needto perform the normalization separately for each feature. 
    
    Hint
    ----
    You might find the 'np.mean' and 'np.std' functions useful.
    """
    # You need to set these values correctly
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])

    # =========================== YOUR CODE HERE =====================
    # Subtract the mean value of each feature from the dataset.   
    for i in range(X.shape[1]):
        mu[i] = np.mean(X[:,i])
        X_norm[:,i] = X[:,i] - mu[i]
     
    # Scale (divide) the feature values by their respective “standard deviations.”
    for i in range(X.shape[1]):
        sigma[i] = np.std(X[:,i])
        X_norm[:,i] = X_norm[:,i] / sigma[i]
    # ================================================================
    return X_norm

