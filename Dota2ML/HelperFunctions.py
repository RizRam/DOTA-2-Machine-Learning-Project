# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 18:56:15 2017

@author: RizMac
"""

import pandas as pd

#Attribute titles
columns = ['gpm', 'xpm', 'kpm', 'dpm', 'apm', 'denpm', 'lhpm', 'spm', 'hdpm',
           'tdpm', 'lpm', 'average team pos', 'duration'] 

#Matches weights to respective attribute name.  Sorts the pairing by weight
#descnending and prints the results.
def printParameterRankings(weights, columns):
    if len(weights) != len(columns):
        print("weights and columns must be of same length")
        return
    
    #weights = abs(weights)
    
    #Create weight column pairs
    weight_column = []
    for i in range(len(columns)):
        weight = weights[i]
        multiplier = 1
        if weight < 0:
            multiplier = -1
        
        weight_column.append((abs(weights[i]), columns[i], multiplier))
    
    #sort by weight in reverse order
    weight_column.sort(key=lambda entry: entry[0], reverse=True)

    #print results
    for i in range(len(weight_column)):
        entry = weight_column[i]
        print('{}: {} ({})'.format(i + 1, entry[1], entry[0] * entry[2]))

#Prints the attribute title in columns based on an ndarray containing index
#numbers
def printColumns(selected, columns):
    if len(selected) > len(columns):
        print("Length of selected must be less than columns")
        return
    
    for i in selected:
        print(columns[i])


#Takes label and binarizes the dataset into 1 and 0 based on threshold value
#If label is greater than or equal to threshold, set the data to 1, else 0
def binarizeLabel(label, threshold):
    #result = label.copy(deep=True);
    
    a = label.as_matrix()
    
    b = a >= threshold
    
    c = b.astype(int)
    
    result = pd.DataFrame(c)
    
    return result
    


    