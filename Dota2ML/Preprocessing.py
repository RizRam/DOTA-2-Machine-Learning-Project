# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 13:52:54 2017

@author: RizMac

Preprocessing Kaggle Dataset
"""


import pandas as pd  #data processing
import os #file check

"""
Proces player-ratings
"""

#process player_ratings.csv
def processPlayerRatings():
    #load player Ratings
    playerRatings = pd.read_csv('../DataSet/player_ratings.csv')
    
    #Remove data account_ids that are negative
    validAccountIDs = playerRatings['account_id'] > 0
    validMatchTotals = playerRatings['total_matches'] >= 50
    
    valid_playerRatings = playerRatings[validAccountIDs & validMatchTotals]
    
    #export to csv
    valid_playerRatings.to_csv('../DataSet/valid_playerRatings.csv', index = False)

"""
Process players.csv
"""
#pre-process players.csv
def processPlayers():
    #delete outputfile if it exists
    filePath = '../DataSet/valid_players.csv'       
    
    if os.path.isfile(filePath):
        os.remove(filePath)
    
    #Load players data in chunks
    
    #create percentage bar
    chunksize = 10 ** 4
    chunks = pd.read_csv('../DataSet/players.csv', chunksize=chunksize)
    totalChunks = len(chunks)
    chunkIndex = 1;
    for chunk in chunks:
        processChunk(chunk, filePath)
        progress = (chunkIndex / totalChunks) * 100
        print("{}%".format(progress))
        chunkIndex += 1

# processes a chunk of data from players.csv
def processChunk(chunk, csv):
    #reduce columns
    columns = ['match_id', 'account_id', 'player_slot', 'gold', 'gold_spent', 'gold_per_min',
               'xp_per_min', 'kills', 'deaths', 'assists', 'denies', 'last_hits', 
               'stuns', 'hero_damage', 'hero_healing', 'tower_damage', 'level',
               'gold_destroying_structure', 'gold_killing_heros', 'gold_killing_creeps',
               ]
    
    filteredData = chunk[columns]
    
    #add desired columns
    filteredData['position'] = 0    
    
    matches = []
    #get player positions
    for i in range(0, len(filteredData), 10):
        match = filteredData[i:i+10]
        determinePlayerPosition(match)
        matches.append(match)
        
    matchesPD = pd.concat(matches)
    filteredData['position'] = matchesPD['position']
    
    #remove accounts_id == 0
    accountCond = filteredData['account_id'] != 0
    validPlayers = filteredData[accountCond]
    
    #write to csv
    if os.path.isfile(csv):
        validPlayers.to_csv(csv, mode='a', index=False, header=False)
    else:
        validPlayers.to_csv(csv, mode='w', index=False)
    
#assign net worth positions to each player in a match    
def determinePlayerPosition(match):          
    
    #get dire and radiant players
    radiantCondition = match['player_slot'] < 5
    direCondition = match['player_slot'] > 127
    radiant = match[radiantCondition]
    dire = match[direCondition]

    #calculate total gold for each player
    radiantGold = []
    for index, row in radiant.iterrows():      
        totalGold = row['gold'] + row['gold_spent']
        radiantGold.append(totalGold)
        
    direGold = []
    for index, row in dire.iterrows():
        totalGold = row['gold'] + row['gold_spent']
        direGold.append(totalGold)
    
    #sort gold values
    radiantGold.sort(reverse=True)
    direGold.sort(reverse=True)
    
    #assign positions
    for i in range(len(radiantGold)):
        posCond = radiant['gold'] + radiant['gold_spent'] == radiantGold[i]
        posIndex = radiant[posCond].index
        radiant.loc[posIndex,'position'] = i + 1
    
    for i in range(len(direGold)):
        posCond = dire['gold'] + dire['gold_spent'] == direGold[i]
        posIndex = dire[posCond].index
        dire.loc[posIndex, 'position'] = i + 1

    radiantDire = [radiant, dire]
    tempPD = pd.concat(radiantDire)
    match['position'] = tempPD['position']

#print contents of valid_players
def checkValidPlayers():
    validPlayers= pd.read_csv('../DataSet/valid_players.csv')
    print(validPlayers)

#edit valid_players.csv 
def editValidPlayers(validPlayers):    
    validPlayers = pd.read_csv('../DataSet/valid_players.csv')    
    validPlayers['stuns'] = pd.to_numeric(validPlayers['stuns'], errors='coerce').fillna(0.0)
    validPlayers['gold_destroying_structure'] = pd.to_numeric(validPlayers['gold_destroying_structure'], downcast='signed').fillna(0)
    validPlayers['gold_killing_heros'] = pd.to_numeric(validPlayers['gold_killing_heros'], downcast='signed')
    validPlayers['gold_killing_creeps'] = pd.to_numeric(validPlayers['gold_killing_creeps'], downcast='signed')
    
    #save edits    
    validPlayers.to_csv('../DataSet/valid_players.csv', mode='w', index=False)
    

"""
Process match.csv
"""

#process match
def processMatch():
    #load match data
    matches = pd.read_csv('../DataSet/match.csv')
    
    #get rid of unwanted attributes
    columns = ['match_id', 'start_time', 'duration', 'first_blood_time', 'radiant_win']
    valid_matches = matches[columns]
        
    #save to csv
    valid_matches.to_csv('../DataSet/valid_matches.csv', index=False)    
    


"""
RUN PROGRAM
"""

processPlayerRatings()
processPlayers()
editValidPlayers()
processMatch()
