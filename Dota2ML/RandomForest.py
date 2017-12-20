# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 22:47:44 2017

@author: RizMac

Random Forest Classifier
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as rcf
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score as fscore
import HelperFunctions as hf


#load data and labels
X = pd.read_csv('../DataSet/normalized_data.csv')  #features
Y = pd.read_csv('../DataSet/valid_label.csv')  #win rate   

#Binarize label
threshold_win_rate = 0.55
Y = hf.binarizeLabel(Y, threshold_win_rate)

##############################################################################
# Find best features
T = 1000 # of trials

best_score = 0.0
best_weights = None 

for i in range(T):
    #Do K-Fold validation 
    k = 10
    kf = KFold(n_splits=k)  
        
    
    
    for train_index, test_index in kf.split(X):
        #get training and test sets
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        Y_train = Y.iloc[train_index]
        Y_test = Y.iloc[test_index]
        
        classifier = rcf()
        classifier.fit(X_train, Y_train.as_matrix().ravel())
        
        Y_predict = classifier.predict(X_test)
        score = fscore(Y_test, Y_predict)
        
        if score > best_score:
            best_score = score
            best_weights = classifier.feature_importances_


###############################################################################
# Print results

print('Random Forest')
print('F-Score: {}'.format(best_score))
hf.printParameterRankings(best_weights, hf.columns)
print('\n')
    
    

