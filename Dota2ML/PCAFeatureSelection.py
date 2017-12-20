# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 16:06:27 2017

@author: RizMac

PCA feature selection
"""

import numpy as np
import pandas as pd


#load data
data = pd.read_csv('../DataSet/normalized_data.csv')
label = pd.read_csv('../DataSet/valid_label.csv')


#column titles
columns = ['gpm', 'xpm', 'kpm', 'dpm', 'apm', 'denpm', 'lhpm', 'spm', 'hdpm',
           'tdpm', 'lpm', 'average team pos', 'duration']  

###############################################################################
# Find largest eigenvalues

#center the data
data = data - data.mean()

#calculate covariance matrix
cov = np.cov(data.T)

#calculate eigenvalues and eigenvectors
eigenValues, eigenVectors = np.linalg.eig(cov)

print(len(eigenValues))

#create eigenValue and columntitle pairs
eigen_column = {}

# {eigenValue : column_index}
for i in range(len(eigenValues)):
    eigen_column[eigenValues[i]] = i
    
#sort eigenValues in descending order
eigenValues = abs(eigenValues)  #get absolute value of eigenvalues
eigenValues = np.sort(eigenValues * -1)  #multiply by -1 to sort descending
eigenValues = abs(eigenValues)  #return to abs value

print('Feature Ranking')
#print results
for i in range(len(eigenValues)):
    ev = eigenValues[i]
    index = eigen_column[ev]
    column = columns[eigen_column[ev]]
    print('{}: {} ({})'.format(i + 1, column, ev))
    


