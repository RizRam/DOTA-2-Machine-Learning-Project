# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 21:48:00 2017

@author: RizMac

Lasso
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
import HelperFunctions as hf



#load data and labels
X = pd.read_csv('../DataSet/normalized_data.csv')  #features
Y = pd.read_csv('../DataSet/valid_label.csv')  #win rate

#Do cross validation
k = 10
model = LassoCV(cv=k)
model.fit(X, Y)

#Get best R2
kf = KFold(n_splits=k)
for train_index, test_index in kf.split(X):
    #get training and test sets
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    Y_train = Y.iloc[train_index]
    Y_test = Y.iloc[test_index]
    
    print(model.score(X_test, Y_test))

hf.printParameterRankings(model.coef_, hf.columns)

