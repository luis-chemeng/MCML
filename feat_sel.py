#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Objetive: Perform the feature selection stage and get the performance per feature curves

Comments:
    - Dataframe of features for dsa :: dfa
    - Dataframe of features for dsb :: dfb

@author: research
"""

# %% Modules

# Basic operation and graph modules
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import dcor

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

# %% Open data

wd = 'RCP2' # Working directory

# If the working directory does not exist, create it
if not os.path.exists(wd):
    os.makedirs(wd)
    print(f"Directory '{wd}' created")

prop = 'RCP'

dsa = pd.read_csv(f'dsa_{prop}.csv')
dsb = pd.read_csv(f'dsb_{prop}.csv')
dfa = pd.read_csv(f'feats_dsa_{prop}.csv')
dfb = pd.read_csv(f'feats_dsb_{prop}.csv')

sep = 517

comp_feats =  dfa.columns[:sep].to_list()

df = dfa

# %% Feature selection correlation

methods = ['pearson', 'spearman', dcor.distance_correlation]

d_methods = dict(zip(['PC', 'SC', 'DC'], methods))

best_corr_preds = dict()

for method_name, method in d_methods.items():
    df_corr = df[comp_feats+[prop]].corr(method=method)
    corrs = df_corr[prop][comp_feats]
    
    idxs_sorted = np.abs(corrs).sort_values(ascending=False).index.to_list()

    idxs_best_nc = []

    for i, idx in enumerate(idxs_sorted):
      if i==0:
        idxs_best_nc.append(idx)
      else:
        if not np.any(np.abs(df_corr[idx][idxs_best_nc]) > 0.9):
          idxs_best_nc.append(idx)

    best_corr_preds.update({method_name+'_decorr' : idxs_best_nc})
    best_corr_preds.update({method_name : idxs_sorted})


# %% Feature selection RFE

X = df[comp_feats]
Y = df[prop]

dict_rfe_best = {}

for i in range(5, 31, 5):
    # Create a decision tree classifier
    regressor = DecisionTreeRegressor(min_samples_leaf=2, random_state=1)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    #X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=67)
    
    # Create an RFE selector
    selector = RFE(regressor, n_features_to_select=i)  # Specify the number of features you want to select
    selector.fit(X_scaled, Y)
    
    # Access the selected features
    selected_features = selector.support_
    
    # Access the importance scores (if the selector exposes them)
    importance_scores = selector.ranking_
    
    # Create a ranking of features based on their importance scores
    ranked_features = [feature for feature, importance in zip(X.columns, importance_scores)]
    
    # Sort the ranked features by importance
    ranked_features.sort(key=lambda x: importance_scores[X.columns.get_loc(x)])
    
    # print("Ranked Features:", ranked_features)
    
    rfe_best = X.columns[selected_features].values
    
    dict_rfe_best.update({i : list(rfe_best)})

# %% Save the feats list

# Save the dictionary to a file
with open(f'{wd}/RFE.json', 'w') as file:
    json.dump(dict_rfe_best, file)
    
with open(f'{wd}/COR.json', 'w') as file:
    json.dump(best_corr_preds, file)


