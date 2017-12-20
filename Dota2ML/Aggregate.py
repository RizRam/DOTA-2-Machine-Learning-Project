# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 14:18:56 2017

@author: RizMac
"""

"""
Combine player and match data
Run this AFTER Preprocessing.py
"""

import pandas as pd  #data processing

MINIMUM_MATCHES = 20

"""
FUNCTIONS
"""

#Determine whether radiant won the match
def getPlayerWin(radiantWin, playerSlot):
    if playerSlot < 5:
        radiant = True
    else:
        radiant = False
    
    return radiantWin == radiant

#Join match table and players table
def joinPlayerMatches():
    matches = pd.read_csv('../DataSet/valid_matches.csv')
    players = pd.read_csv('../DataSet/valid_players.csv')
    
    players['start_time'] = 0
    players['duration'] = 0
    players['win'] = True
    
    current = 1.0;
    for index, row in matches.iterrows():
        matchCond = players['match_id'] == row['match_id']
        idx = players[matchCond].index
        
        players.loc[idx, 'start_time'] = row['start_time']
        players.loc[idx, 'duration'] = row['duration']
        players.loc[idx, 'win'] = row['radiant_win'] == (players.loc[idx, 'player_slot'] < 5)
        
        if current % 100 == 0:
            print("{:.2f}%".format(((current / len(matches) * 100))))              
            
        current += 1.0
    
    #write to file
    players.to_csv('../DataSet/valid_player_match.csv', index=False, mode='w')

#Aggregate players from multiple matches
def aggregatePlayers():
    playerMatches = pd.read_csv('../DataSet/valid_player_match.csv')
    
    #group rows by account_id
    playerList = []    
    aggregatePlayers = playerMatches.groupby('account_id')
    for id, group in aggregatePlayers:
        matchesPlayed = len(group)
        if matchesPlayed >= MINIMUM_MATCHES:
            #aaggregate data
            gold = group['gold'].mean() + group['gold_spent'].mean()            
            gpm = group['gold_per_min'].mean()
            xpm = group['xp_per_min'].mean()
            kills = group['kills'].mean()
            deaths = group['deaths'].mean()
            assists = group['assists'].mean()
            denies = group['denies'].mean()
            lastHits = group['last_hits'].mean()
            stuns = group['stuns'].mean()
            heroDamage = group['hero_damage'].mean()
            towerDamage = group['tower_damage'].mean()
            level = group['level'].mean()
            goldStructure = group['gold_destroying_structure'].mean()
            goldHeros = group['gold_killing_heros'].mean()
            goldCreeps = group['gold_killing_creeps'].mean()
            position = group['position'].mean()
            duration = group['duration'].mean()
            winRate = group['win'].mean()
            gamesPlayed = len(group)
            
            dataPoint = [id, gold, gpm, xpm, kills, deaths, assists, denies, 
                         lastHits, stuns, heroDamage, towerDamage, level, 
                         goldStructure, goldHeros, goldCreeps, position, 
                         duration, winRate, gamesPlayed]            
            playerList.append(dataPoint)
    
    #create DataFrame
    columns = ['id', 'gold', 'gpm', 'xpm', 'kills', 'deaths', 'assists', 
               'denies', 'lastHits', 'stuns', 'hero damage', 'tower damage', 
               'level', 'gold from structure', 'gold from hero kills', 
               'gold from creeps', 'average team position', 'duration', 
               'win rate', 'total matches']
    
    playerFrame = pd.DataFrame(playerList, columns=columns)    
    
    #save DataFrame to csv
    playerFrame.to_csv('../DataSet/valid_player_dataset.csv', index=False, mode='w')

#check contents of valid_player_match    
def checkPlayerMatches():
    playerMatches = pd.read_csv('../DataSet/valid_player_match.csv')
    print(playerMatches[:3])

#check contents of valid_player_dataset
def checkValidPlayerDataset():
    dataset = pd.read_csv('../DataSet/valid_player_dataset.csv')
    print(dataset[:3])
    print(dataset.shape)
    



"""
RUN PROGRAM
"""

#checkPlayerMatches()
#aggregatePlayers()
checkValidPlayerDataset()
