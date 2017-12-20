# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 17:19:41 2017

@author: RizMac

Regression Parameters
"""


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, LassoCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score as r2
import HelperFunctions as hf



#load data and labels
X = pd.read_csv('../DataSet/normalized_data.csv')  #features
Y = pd.read_csv('../DataSet/valid_label.csv')  #win rate

############################################################################
#Get best regression model

#Do k-fold validation
k = 10
kf = KFold(n_splits=k)

#Save best model
lr_best_score = 0.0
lr_best_weights = None
ridge_best_score = 0.0
ridge_best_weights = None
lasso_best_score = 0.0
lasso_best_weights = None

for train_index, test_index in kf.split(X):   
    
    #get training and test sets
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    Y_train = Y.iloc[train_index]
    Y_test = Y.iloc[test_index]
    
    #create models
    lr = LinearRegression()
    lr.fit(X_train, Y_train)
    
    ridge = Ridge()
    ridge.fit(X_train, Y_train)
    
    lasso = LassoCV()
    lasso.fit(X_train, Y_train.as_matrix().ravel())
    
    #get predicted labels
    lr_predict = lr.predict(X_test)
    ridge_predict = ridge.predict(X_test)   
    lasso_predict = lasso.predict(X_test)
    
    #Save best trials
    lr_score = r2(Y_test, lr_predict)
    if lr_score > lr_best_score:
        lr_best_score = lr_score
        lr_best_weights = lr.coef_[0]
    
    ridge_score = r2(Y_test, ridge_predict)
    if ridge_score > ridge_best_score:
        ridge_best_score = ridge_score
        ridge_best_weights = ridge.coef_[0]
    
    lasso_score = r2(Y_test, lasso_predict)    
    if lasso_score > lasso_best_score:
        lasso_best_score = lasso_score
        lasso_best_weights = lasso.coef_


#############################################################################
# Print results

print('Least Squares Regression')
print('R2: {}'.format(lr_best_score))
hf.printParameterRankings(lr_best_weights, hf.columns)

print('\n')
print('Ridge Regression')
print('R2: {}'.format(ridge_best_score))
hf.printParameterRankings(ridge_best_weights, hf.columns)

print('\n')
print('Lasso')
print('R2: {}'.format(lasso_best_score))
hf.printParameterRankings(lasso_best_weights, hf.columns)

    


    