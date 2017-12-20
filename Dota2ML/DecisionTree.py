# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 18:36:29 2017

@author: RizMac
Decision Tree classifier
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier as tree
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score as fscore
import HelperFunctions as hf


#load data and labels
X = pd.read_csv('../DataSet/normalized_data.csv')  #features
Y = pd.read_csv('../DataSet/valid_label.csv')  #win rate   

#Binarize label
threshold_win_rate = 0.55
Y = hf.binarizeLabel(Y, threshold_win_rate)

###############################################################################
# Find best classifier

T = 1000 #of trials 

best_score = 0.0
best_features = None
for i in range(T):
    #K fold validation
    k = 10
    kf = KFold(n_splits=k)   
    
    for train_index, test_index in kf.split(X):
        #get training and test sets
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        Y_train = Y.iloc[train_index]
        Y_test = Y.iloc[test_index]
    
        classifier = tree()
        classifier.fit(X_train, Y_train)
        
        #get predicted values
        Y_predict = classifier.predict(X_test)
        score = fscore(Y_test, Y_predict)  #get score
        
        #save classifier weights for best score
        if score > best_score:
            best_score = score
            best_features = classifier.feature_importances_

###############################################################################
# Print results

print('Decision Tree')
print('F-Score: {}'.format(best_score))
hf.printParameterRankings(best_features, hf.columns)
print('\n')
    
    