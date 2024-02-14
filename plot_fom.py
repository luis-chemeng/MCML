#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 21:05:14 2024

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


# %% Load predictions of best models

prop2 = 'RCP'

d_p = {
       'RCP' : 'RCP [J/Kg]',
       'ME' : 'ME [J/(KgK)]'
       }

prop2_units = d_p[prop2]

rcp_predicts = pd.read_csv('preditc_RCP4_XGB_A_MinDelta.csv', index_col=0)
tc_predicts = pd.read_csv('preditc_TC4_KRR_A_Var.csv', index_col=0)
me_predicts = pd.read_csv('preditc_ME4_XGB_A_RFE10.csv', index_col=0)
df_elems = pd.read_csv('df_new_with_elems.csv', index_col=0)

if prop2 == 'RCP':
    predicts = tc_predicts.merge(rcp_predicts, on='Comp')
else:
    predicts = tc_predicts.merge(me_predicts, on='Comp')
    
predicts = predicts.merge(df_elems, on='Comp')

# %% 

lims = [20, 430]
step = 0.5
top = 1



colors = ['tab:blue', 'tab:red', 'tab:green']


intervals = np.arange(lims[0], lims[1], step)


fig, ax = plt.subplots()


# for i, A in enumerate(['La', 'Nd', 'Pr']):
#     dfi = predicts[predicts[A]>=0.6]
#     ax.scatter(dfi['TC'].values, dfi['RCP'].values, s=10, c=colors[i], alpha=0.05,)


df_max_values = []    
for intv in intervals:
    dfi = predicts[(predicts['TC']>=intv)&(predicts['TC']<intv+5)].sort_values(by=prop2).iloc[-1*top:]
    df_max_values.append(dfi)

df_max_values = pd.concat(df_max_values)

for i, A in enumerate(['La', 'Nd', 'Pr']):
    dfi = df_max_values[df_max_values[A]>=0.6]  
    ax.scatter(dfi['TC'].values, dfi[prop2].values, s=15, c=colors[i], alpha=0.8, label=f'A site : {A}', zorder=1)


special_lims = [288, 308]
bps = predicts[(predicts['TC']>=special_lims[0])&(predicts['TC']<special_lims[1])].sort_values(by=prop2).iloc[-3:] 
ax.scatter(bps['TC'].values, bps[prop2].values, s=15, c='tab:blue', alpha=0.8, zorder=4)

arrowprops = dict(facecolor='none', edgecolor='black', arrowstyle='->', shrinkA=0, linewidth=1.5)

pos = {
       'ME' : [[180, 2.7], [250, 2.2], [320, 4.5]],
       'RCP' : [[316, 135], [315, 147], [280, 160]],
       }

for i in range(3):
    s = u.format_str_to_latex(bps['Comp'].iloc[i])
    ax.annotate(
        s,  # Text of the label
        xy=(bps['TC'].iloc[i], bps[prop2].iloc[i]),  # Coordinates of the point
        xytext=(pos[prop2][i][0], pos[prop2][i][1]),  # Coordinates of the text (adjust as needed)
        arrowprops=arrowprops,  # Arrow properties
        fontsize=7,  # Font size of the label
        color='black',  # Color of the label
        weight='bold',  # Font weight of the label
        ha='left',  # Horizontal alignment of the text
        va='center',  # Vertical alignment of the text
    )
    # ax.text(bps['TC'].iloc[i]+pos[prop2][i][0], bps[prop2].iloc[i]+pos[prop2][i][1], s=s, fontsize=7)
    ax.scatter(bps['TC'].iloc[i], bps[prop2].iloc[i], edgecolor='k', s=25, color='white', zorder=3)



for line in special_lims:
    ax.axvline(x=line, color='r', linestyle='--', linewidth=1, zorder=0)

if prop2 == 'RCP':
    ax.scatter(295, 128, s=15, c='tab:orange', alpha=0.8, label='Gd') # Data from : 10.1088/1402-4896/abc984
else:
    ax.scatter(295, 5.54, s=15, c='tab:orange', alpha=0.8, label='Gd') # Data from : 10.12693/APhysPolA.128.111

ax.grid(alpha=0.7, linestyle=':', zorder=0)
ax.set_xlabel('Tc [K]')
ax.set_ylabel(prop2_units)
ax.set_xlim([lims[0], lims[1]])
if prop2 == 'RCP':
    ax.set_ylim([85, 205])
else:
    pass
ax.legend()

plt.tight_layout()
fig.savefig(f'fom_{prop2}', dpi=300)

