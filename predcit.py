#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Objetive: Predict the properties of the New Potential poerovskites with the best models

@author: research
"""

# %% Modules

# Basic operation and graph modules
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Auxiliary modules
import utilities as u
import os
import time

# Machine learning modules
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, Trials

# More
import json
import joblib

# %% Open data

prop = 'ME'

wd = 'ME4' # Working directory

dsa = pd.read_csv(f'dsa_{prop}.csv')
dsb = pd.read_csv(f'dsb_{prop}.csv')
dfa = pd.read_csv(f'feats_dsa_{prop}.csv')
dfb = pd.read_csv(f'feats_dsb_{prop}.csv')

comp_feats =  dfa.columns[:517].to_list()

with open(f'{wd}/RFE.json', 'r') as file:
    dict_rfe_best = json.load(file)

with open(f'{wd}/COR.json', 'r') as file:
    dict_cor_best = json.load(file)
    
dict_group_feats = {
    'All517' : comp_feats[:517],
    'Sum' : comp_feats[:58],
    'Mean' : comp_feats[58:58*2],
    'Var' : comp_feats[58*2:58*3],
    'Geo' : comp_feats[58*3:58*4],
    'Har' : comp_feats[58*4:58*5],
    'Min' : comp_feats[58*5:58*6],
    'Max' : comp_feats[58*6:58*7],
    'TolF' : comp_feats[58*7:459],
    'MinDelta' : comp_feats[459:517],
    }

n_feats = ['10', '20', '30']

for n_feat in n_feats:
    dict_group_feats.update({'SC'+n_feat : dict_cor_best['SC_decorr'][:int(n_feat)]})
    dict_group_feats.update({'PC'+n_feat : dict_cor_best['PC_decorr'][:int(n_feat)]})
    dict_group_feats.update({'DC'+n_feat : dict_cor_best['DC_decorr'][:int(n_feat)]})
    dict_group_feats.update({'RFE'+n_feat : dict_rfe_best[n_feat]})


# %% Load model

dataset = 'A'

best_models = {
    'TC' : [['KRR', 'Var'],['KRR', 'RFE10']],
    'ME' : [['XGB', 'RFE10'],['KRR', 'RFE10']],
    'RCP' : [['XGB', 'MinDelta'],['XGB', 'DC10']],
    }

if dataset=='A':
    model = best_models[prop][0][0]
    feats = best_models[prop][0][1]
else:
    model = best_models[prop][1][0]
    feats = best_models[prop][1][1]


model1 = joblib.load(f'{wd}/{model}_{dataset}_{feats}.joblib')


pred = dict_group_feats[feats]

if dataset=='A':
    df = dfa
    lucky_seed = 42
else:
    df = dfb
    lucky_seed = 42

if prop=='TC':
    if dataset=='A':
        X_sub = df[pred]
    else:
        X_sub = df[pred+['D']]
else:
    if dataset=='A':
        X_sub = df[pred+['H']]
    else:
        X_sub = df[pred+['H','D']]
        
Y_sub = np.array(df[prop])

scaler = StandardScaler()
X_sub_scaled = scaler.fit_transform(X_sub)

X_train, X_test, Y_train, Y_test = train_test_split(X_sub_scaled, Y_sub, test_size=0.2, random_state=lucky_seed)

u.parity_plot(model1, X_train, X_test, Y_train, Y_test)



# %% Load New Comps.

df_T = pd.read_csv('df_new_comps_featurized.csv')

with open('new_comps.json', 'r') as file:
    new_comps = json.load(file)

df_new = pd.DataFrame(new_comps, columns=['Comp'])


# %% Magnetic Field and Crystalite effect, if requeried

H = 2
D = 40

if prop!='TC':
    df_T['H'] = H
    pred = pred + ['H']

if dataset=='B':
    df_T['D'] = D
    pred = pred + ['D']

# %% Predict and save the predictions

X_new = df_T[pred]  
    
X_new_scaled = scaler.transform(X_new)

Y_new_pred = model1.predict(X_new_scaled)

df_new[prop] = Y_new_pred 

df_new.to_csv(f'preditc_{wd}_{model}_{dataset}_{feats}.csv')  


  
