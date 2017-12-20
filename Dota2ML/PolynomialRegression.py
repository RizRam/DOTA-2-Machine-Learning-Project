# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 18:36:46 2017

@author: RizMac

Polynomial Regression
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse
import HelperFunctions as hf

#load data and labels
X = pd.read_csv('../DataSet/normalized_data.csv')  #features
Y = pd.read_csv('../DataSet/valid_label.csv')  #win rate

#degrees
#degrees = np.arange(1, 5, 1)
degrees = [3]

#test multiple polynomial degrees
for degree in degrees:
    pipeLine = Pipeline([('poly', PolynomialFeatures(degree=degree)),
                         ('linear', LinearRegression())])
    
    #Do k-fold validation
    k = 10
    kf = KFold(n_splits=k)
    
    best_score = 0.0
    best_weights = None
    
    for train_index, test_index in kf.split(X):    
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        Y_train = Y.iloc[train_index]
        Y_test = Y.iloc[test_index]
        
        #train model
        pipeLine.fit(X_train, Y_train)
        
        #get prediction
        Y_predict = pipeLine.predict(X_test)
        
        score = mse(Y_test, Y_predict)
        
        if score > best_score:
            best_score = score
            best_weights = pipeLine.named_steps['linear'].coef_[0]
    
    print('degree:{} score: {}'.format(degree, best_score))
    print(best_weights)
    hf.printParameterRankings(best_weights, hf.columns)
        
        
        
    

