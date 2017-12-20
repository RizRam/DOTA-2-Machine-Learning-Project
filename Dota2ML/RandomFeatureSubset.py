# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 22:13:13 2017

@author: RizMac

Random Feature Subset
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as lr
from sklearn.model_selection import KFold

from sklearn.metrics import f1_score as fscore
import HelperFunctions as hf



#load data and labels
X = pd.read_csv('../DataSet/normalized_data.csv')  #features
Y = pd.read_csv('../DataSet/valid_label.csv')  #win rate   

threshold = 0.55
Y = hf.binarizeLabel(Y, threshold)
    

total_features = X.shape[1]
number_feature_subsets = range(1, total_features)
T = 5000  #of trials


for k in number_feature_subsets:
    FSel = []  #subset of features
    best_score = 0.0
    for seed in range(1, T + 1):
        np.random.seed(seed)
        rp = np.random.permutation(total_features)
        
        features = rp[:k]
        
        dataset = X.as_matrix()
        dataset = dataset[:,features]
        
        folds = 10
        kf = KFold(n_splits=folds)
        scores = []
        for train_index, test_index in kf.split(dataset):
            #get training and test sets
            X_train = dataset[train_index,:]
            X_test = dataset[test_index,:]
            Y_train = Y.iloc[train_index]
            Y_test = Y.iloc[test_index]
            
            model = lr(penalty='l2', solver='liblinear')
            model.fit(X_train, Y_train)
            
            Y_pred = model.predict(X_test)        
            score = fscore(Y_test, Y_pred)
            
            scores.append(score)        
        
        meanScore = np.mean(scores)        
        
        if meanScore > best_score:
            best_score = meanScore
            FSel = features
        
    
    print('\n')
    print('k: {}'.format(k))
    print('F-score: {}'.format(best_score))
    hf.printColumns(FSel, hf.columns)
    
    
    
        
    
        
        
        
        
        
