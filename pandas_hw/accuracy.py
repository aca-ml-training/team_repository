import numpy as np
import pandas as pd
import cmath
from sklearn.linear_model import LogisticRegression


def loss(Y,Y_pred):
    """
    loss - a function evaluating the loss of the obtained result
    
    :param Y: 1D numpy array of floats representing the correct labels
    :param Y_pred: 1D numpy array of floats representing the predicted labels
    
    """

    Y = Y.tolist()
    Y_pred = Y_pred.tolist()
    score = 0
    for i in range(len(Y)):
        score += (Y[i]-Y_pred[i])**2
    score=cmath.sqrt(score/len(Y))
    return score

def cross_validate(X, Y, folds=5):
    
    """
    cross_validate - a function performing cross validation for a specified number of folds and calculating 
    the mean of accuracies obtained during each partitioning
    
    :param X: 2D numpy array representing the data
    :param Y: 1D numpy array representing the labels
    :param folds: an integer representing the number of cross-validation folds
    
    """

    log = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
    intercept_scaling=1, max_iter=200, multi_class='ovr', n_jobs=3,
    penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
    verbose=0, warm_start=False)
        

    


    scores_log = [] 
    scores_forest = []
    index = np.arange(X.shape[0])
    score_log = 0
    score_forest = 0
    
    for i in range(folds):
        score_log = 0
        score_forest = 0
        
        test_index = np.random.choice(index, int(X.shape[0]*1/folds),replace=False)
        index = np.setdiff1d(np.arange(X.shape[0]),test_index)
      
        test_x = X[test_index]
        test_y = Y[test_index]

        log.fit(X[index],Y[index])
        pred_log = log.predict(test_x)
        
        ran.fit(X[index],Y[index])
        pred_ran = ran.predict(test_x)
        
        for i in range(len(test_y)):
            if(pred_log[i] == test_y[i]):
                score_log += 1
            if(pred_ran[i] == test_y[i]):
                score_forest += 1
        scores_log.append(score_log/len(test_y))
        scores_forest.append(score_forest/len(test_y))
        

    return (np.mean(scores_log),np.mean(scores_forest))

