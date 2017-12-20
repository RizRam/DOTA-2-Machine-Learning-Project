# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 14:57:02 2017

@author: RizMac
Normalize Data
Run AFTER Aggregate.py and Preprocessing.py
"""

import pandas as pd  #data processing

columns = {}
columns[0] = 'id'
columns[1] = 'gold'
columns[2] = 'gpm'
columns[3] = 'xpm'



data = pd.read_csv('../DataSet/valid_player_dataset.csv')

#express some columns in terms of match duration
def convertToPM(series):
    return series / (data['duration'] / 60.0)

kpm = convertToPM(data['kills'])  #kills per min
dpm = convertToPM(data['deaths'])  #deaths per min
apm = convertToPM(data['assists'])  #assists per min
denpm = convertToPM(data['denies'])  #denies per min
lhpm = convertToPM(data['lastHits'])  #last hits per min
spm = convertToPM(data['stuns'])  #stuns per min
hdpm = convertToPM(data['hero damage'])  #hero damage per min
tdpm = convertToPM(data['tower damage'])  #tower damage per min
lpm = convertToPM(data['level'])  #level per min

#Get wanted columns
data_norm = pd.DataFrame()
data_norm['gpm'] = data['gpm']
data_norm['xpm'] = data['xpm']
data_norm['kpm'] = kpm
data_norm['dpm'] = dpm
data_norm['apm'] = apm
data_norm['denpm'] = denpm
data_norm['lhpm'] = lhpm
data_norm['spm'] = spm
data_norm['hdpm'] = hdpm
data_norm['tdpm'] = tdpm
data_norm['lpm'] = lpm
data_norm['average team pos'] = data['average team position']
data_norm['duration'] = data['duration']

#Get label
columns = ['win rate']
label = pd.DataFrame(data['win rate'], columns=columns)
print(label)
label.to_csv('../DataSet/valid_label.csv', index=False)

##############################################################################
# Normalize Data

def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

data_norm['gpm'] = normalize(data_norm['gpm'])
data_norm['xpm'] = normalize(data_norm['xpm'])
data_norm['kpm'] = normalize(data_norm['kpm'])
data_norm['dpm'] = normalize(data_norm['dpm'])
data_norm['apm'] = normalize(data_norm['apm'])
data_norm['denpm'] = normalize(data_norm['denpm'])
data_norm['lhpm'] = normalize(data_norm['spm'])
data_norm['spm'] = normalize(data_norm['spm'])
data_norm['hdpm'] = normalize(data_norm['hdpm'])
data_norm['tdpm'] = normalize(data_norm['tdpm'])
data_norm['lpm'] = normalize(data_norm['lpm'])
data_norm['average team pos'] = normalize(data_norm['average team pos'])
data_norm['duration'] = normalize(data_norm['duration'])

#save to csv
data_norm.to_csv('../DataSet/normalized_data.csv', index=False)






