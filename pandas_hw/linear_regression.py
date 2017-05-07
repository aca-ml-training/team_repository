#!/usr/bin/env python3
"""
Run regression on apartment data.
"""
from __future__ import print_function
import argparse
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import getpass




def featurize(apartment):

    """
    featurize - a function taking a dataframe row and returning a tuple with (x,y),
    where x and y are the feature vector and the correspoding label respectively
    
    :param apartment: Apartment DataFrame row (a dictionary like object)
    :return: (x, y) tuple, where x is a numpy vector, and y is a number
    
    """
    NewlyRepaired = Good = Zero = Center = Other = 0

    if(apartment["district"] == 'Center'):
        Center = 1      
    else:
        Other = 1
        
    if(apartment["condition"] == 'good'):
        Good = 1
    if(apartment["condition"] == 'newly repaired'):
        NewlyRepaired = 1
    if(apartment["condition"] == 'zero condition'):
        Zero = 1
     
    x = np.array([1,Other,apartment["max_floor"],apartment["floor"],Good,apartment["ceiling_height"],
                  apartment["num_bathrooms"],NewlyRepaired,Center,apartment["num_rooms"],apartment["area"],
                  apartment["area"]/apartment["num_rooms"]])
    
    return x, apartment['price']


def fit_ridge_regression(X, Y, l=1.4):
    

    """
    :param X: A numpy matrix, where each row is a data element (X)
    :param Y: A numpy vector of responses for each of the rows (y)
    :param l: ridge variable
    :return: A vector containing the hyperplane equation (beta)
    
    """
  
  
    betaHat = np.dot(np.dot(np.linalg.inv(np.matrix(X.T.dot(X)) + l*np.identity(X.shape[1])),X.T),np.array([Y]).T)
    
    return betaHat


def cross_validate(X, Y, fitter, folds=5):
    
    """
    cross_validate - a function performing cross validation for a specified number of folds and calculating 
    the mean of accuracies obtained during each partitioning
    
    :param X: A numpy matrix, where each row is a data element (X)
    :param Y: A numpy vector of responses for each of the rows (y)
    :param fitter: A function that takes X, Y as parameters and returns beta
    :param folds: number of cross validation folds (parts)
    :return: list of corss-validation scores
    
    """
    
    import cmath
    scores = [] 
    index = np.arange(X.shape[0])
    
    for i in range(folds):
        
        test_index = np.random.choice(index, int(X.shape[0]*1/folds),replace=False)
        index = np.setdiff1d(np.arange(X.shape[0]),test_index)
      
        test_x = X[test_index]
        test_y = Y[test_index]

        beta = fitter(X[index],Y[index])

        score = 0
 
        for i in range(len(test_y)):
            score += (test_y[i]-np.dot(beta.transpose(),test_x[i]))**2
            pass
     
        scores.append(cmath.sqrt(score/len(test_y)))

    return scores




def my_beta():

    """
    my_beta - a funciton taking no arguments and returning a beta value obtained by fitting ridge regression
    
    """
    return np.array([-77156.11306457, -57488.31537804, 138.13872621,-128.85820253, 9362.7089974, 39426.21684615,
        16519.0178405 , 23782.09233161, -19667.79768369, -20107.37409211, 1626.00462753, -771.89429512])
