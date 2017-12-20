# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 19:10:49 2017

@author: RizMac
SVM
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
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
# Get best model

#Do K-Fold validation
k = 10
kf = KFold(n_splits = 10)

best_fscore = 0.0
best_fs_weights = None

for train_index, test_index in kf.split(X):
    #get training and test sets
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    Y_train = Y.iloc[train_index]
    Y_test = Y.iloc[test_index]
    
    classifier = SVC(kernel='linear')
    classifier.fit(X_train, Y_train)
    
    Y_predict = classifier.predict(X_test)
    
    fs = fscore(Y_test, Y_predict)
    
    if fs > best_fscore:
        best_fscore = fs
        best_fs_weights = classifier.coef_[0]

#Print results
print('Linear SVM')
print('F-Score: {}'.format(best_fscore))
hf.printParameterRankings(best_fs_weights, hf.columns)
print('\n')   

    
    