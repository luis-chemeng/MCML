#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 21:52:51 2024

@author: research
"""

# %% Modules

# Basic operation and graph modules
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import dcor
import seaborn as sns

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

# More
import json
import joblib

# %% 


prop = 'RCP'

wd = 'RCP4' # Working directory

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
    # 'All406' : comp_feats[:58*7],
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

datasets = ['A', 'B']

best_models = {
    'TC' : [['KRR', 'Var'],['KRR', 'RFE10']],
    'ME' : [['XGB', 'RFE10'],['KRR', 'RFE10']],
    'RCP' : [['XGB', 'MinDelta'],['XGB', 'Geo']],
    }

d_prop2 = {
    'TC' : 'Tc [K]',
    'ME' : 'ME [J/(kg*K)]',
    'RCP' : 'RCP [J/kg]',
}

best_model = best_models[prop]

fig, ax = plt.subplots(2, len(datasets), figsize=(9,7))

for i, dataset in enumerate(datasets):

    model = best_model[i][0]
    feats = best_model[i][1]
    
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
    
    y_pred_train = model1.predict(X_train)
    y_pred_test = model1.predict(X_test)
    
    ax[0,i] = u.parity_plot_(ax[0,i], y_pred_train, y_pred_test, Y_train, Y_test, error_lines=[-0.2, 0.2])
    label = d_prop2[prop]
    ax[0,i].set_xlabel(f'Exp. {label}')
    ax[0,i].set_ylabel(f'Pred. {label}')
    
    ax[1,i] = u.histogram_residuals_(
        ax[1,i], 
        y_pred_train, 
        y_pred_test, 
        Y_train, 
        Y_test, 
        bins=11,
        )
    
    ax[1,i].set_xlabel(r'$y_{Pred.} - y_{Exp.}$')
    ax[1,i].set_ylabel('Density')
    ax[0,i].set_title(f'Dataset {dataset} - {model} - {feats}')

    ax[0,i].legend()    


plt.tight_layout()
fig.savefig(f'parity_{prop}', dpi=300)    

# %% 





