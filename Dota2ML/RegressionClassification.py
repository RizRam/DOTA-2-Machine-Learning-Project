# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 17:51:45 2017

@author: RizMac

Regression Classification
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as lr
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score as fscore
import HelperFunctions as hf


#Load data
#load data and labels
X = pd.read_csv('../DataSet/normalized_data.csv')  #features
Y = pd.read_csv('../DataSet/valid_label.csv')  #win rate   

#Binarize label
threshold_win_rate = 0.55
Y = hf.binarizeLabel(Y, threshold_win_rate)


###############################################################################
# Get best model

penalty_values = ['l1', 'l2']  #regularization strategies

for penalty in penalty_values:
    
    #K fold validation
    k = 10
    kf = KFold(n_splits=k)
    
    
    best_score = 0.0
    best_weights = 0.0
    for train_index, test_index in kf.split(X):
        #get training and test sets
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        Y_train = Y.iloc[train_index]
        Y_test = Y.iloc[test_index]
        
        #train classifier
        classifier = lr(penalty=penalty, solver='liblinear')
        classifier.fit(X_train, Y_train)
        
        #get predicted values
        Y_predict = classifier.predict(X_test)
        score = fscore(Y_test, Y_predict)  #get score
        
        #save classifier weights for best score
        if score > best_score:
            best_score = score
            best_weights = classifier.coef_[0]
    
    # Print results
            
    print('Logistic Regression ({})'.format(penalty))
    print('FScore: {}'.format(best_score))
    hf.printParameterRankings(best_weights, hf.columns)
    print('\n')