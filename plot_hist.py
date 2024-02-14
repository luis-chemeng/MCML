#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 12:26:33 2024

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

d_prop2 = {
    'TC' : 'Tc [K]',
    'ME' : 'ME [J/(kg*K)]',
    'RCP' : 'RCP [J/kg]',
}

Ya = []
Yb = []

for prop in d_prop2.keys():
    dsa = pd.read_csv(f'dsa_{prop}.csv')
    dsb = pd.read_csv(f'dsb_{prop}.csv')
    Ya.append(dsa['Value'].values)
    Yb.append(dsb['Value'].values)
    

fig, ax = plt.subplots(1, 3, sharey=True, figsize=(12, 5))

for i, prop in enumerate(d_prop2.keys()):
    # Create a histogram
    
    bins = np.histogram_bin_edges(Ya[i], bins='auto')
    # bins = np.histogram_bin_edges(Ya[i], bins='scott')
    
    ax[i].hist(Ya[i], bins=bins, color='tab:blue', edgecolor='black', linewidth=0.8, zorder=98, alpha=0.8, label='Dataset A')#range=(Ys[i].min(), Ys[i].max()), alpha=1, zorder=10, label='Without D')
    ax[i].hist(Yb[i], bins=bins, color='tab:green', edgecolor='black', linewidth=0.8, zorder=99, alpha=0.8, label='Dataset B')#range=(Ys[i].min(), Ys[i].max()), alpha=1, zorder=10, label='With D')
    ax[i].set_xlabel(d_prop2[prop])
    ax[i].grid(zorder=1)
    ax[i].text(0.4, 0.62, str(len(Ya[i])), transform=ax[i].transAxes, color='tab:blue', fontsize=12, weight='bold',)
    ax[i].text(0.4, 0.55, str(len(Yb[i])), transform=ax[i].transAxes, color='tab:green', fontsize=12, weight='bold',)

ax[0].set_ylabel('Count')
ax[-1].legend()

plt.tight_layout()
fig.savefig('Hist', dpi=300)
