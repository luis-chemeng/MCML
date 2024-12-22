#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 11:56:43 2024

@author: luchorsh
"""

from matplotlib import rcParams
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

rcParams['font.family'] = 'sans-serif'
rcParams['font.serif'] = ['Helvetica'] + rcParams['font.serif']
plt.rcParams.update({'font.size': 14})

files_d ={
    'TC' : [['predict_TC10_KRR_A_Var.csv'], ['predict_TC10_KRR_B_RFE10_D40.csv', 'predict_TC10_KRR_B_RFE10_D80.csv']],
    'ME' : [['predict_ME12_XGB_A_RFE10.csv'], ['predict_ME12_KRR_B_RFE10_D40.csv', 'predict_ME12_KRR_B_RFE10_D80.csv']],
    'RCP' : [['predict_RCP12_XGB_A_MinDelta.csv'], ['predict_RCP12_XGB_B_Geo_D40.csv', 'predict_RCP12_XGB_B_Geo_D80.csv']],
    }

d_p = {
       'ME' : r'$ME \: [\mathrm{J \: kg^{-1} \: K^{-1}}]$',
       'RCP' : r'$\it{RCP} \: [\mathrm{J \: kg^{-1}}]$',
       'TC' : r'$T_C$ [K]'
       }


elems = pd.read_csv('df_new_with_elems.csv', index_col=0)


data_tc = pd.read_csv(files_d['TC'][0][0], index_col=0)
data_p1 = pd.read_csv(files_d['ME'][0][0], index_col=0)
data_p2 = pd.read_csv(files_d['RCP'][0][0], index_col=0)
data = pd.merge(data_tc, data_p1, on='Comp')
data = pd.merge(data, data_p2, on='Comp')
data = pd.merge(data, elems, on='Comp')


# %%

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

A_sites = ['La', 'Pr', 'Nd']
colors = ['tab:blue', 'tab:green', 'tab:red']

step = 1
top = 1

ts = np.arange(40, 400, step)

# Contour
df_max_values_ME = []    
for t in ts:
    dfi = data[(data['TC']>=t)&(data['TC']<t+step)].sort_values(by='ME')
    if dfi.shape[0] > 0:
        df_max_values_ME.append(dfi.iloc[-1*top:])

df_max_values_ME = pd.concat(df_max_values_ME)

for i, A in enumerate(A_sites):
    dfi = df_max_values_ME[df_max_values_ME[A]>=0.6]  
    ax[0].scatter(dfi['TC'].values, dfi['ME'].values, s=15, c=colors[i], alpha=0.8, label=f'A site: {A}', zorder=1, marker='o')

df_max_values_RCP = []    
for t in ts:
    dfi = data[(data['TC']>=t)&(data['TC']<t+step)].sort_values(by='RCP')
    if dfi.shape[0] > 0:
        df_max_values_RCP.append(dfi.iloc[-1*top:])

df_max_values_RCP = pd.concat(df_max_values_RCP)

for i, A in enumerate(A_sites):
    dfi = df_max_values_RCP[df_max_values_RCP[A]>=0.6]  
    ax[1].scatter(dfi['TC'].values, dfi['RCP'].values, s=15, c=colors[i], alpha=0.8, label=f'A site: {A}', zorder=1, marker='o')



# Gadolinium Ref Data 
gd_TC = np.array([295, 292, 295.7])
gd_RCP = np.array([124.8, 172.2, 230.6])
gd_ME = np.array([4.28, 4.17, 5.57])
ax[1].errorbar(gd_TC.mean(), gd_RCP.mean(), xerr=gd_TC.std(), yerr=gd_RCP.std(), c='tab:orange', alpha=0.8, label='Gd - Ref. Data', marker='.', capsize=3, elinewidth=1, linestyle='')
ax[0].errorbar(gd_TC.mean(), gd_ME.mean(), xerr=gd_TC.std(), yerr=gd_ME.std(), c='tab:orange', alpha=0.8, label='Gd - Ref. Data', marker='.', capsize=3, elinewidth=1, linestyle='')  

# Limits
ylims = ax[0].get_ylim()
ax[0].set_ylim([1.98, ylims[1]])
xlims = ax[0].get_xlim()
ax[0].set_xlim([40, xlims[1]])

ylims = ax[1].get_ylim()
ax[1].set_ylim([66, ylims[1]])
xlims = ax[1].get_xlim()
ax[1].set_xlim([40, xlims[1]])


# Labels
ax[0].set_xlabel(d_p['TC'])
ax[1].set_xlabel(d_p['TC'])
ax[0].set_ylabel(d_p['ME'])
ax[1].set_ylabel(d_p['RCP'])
ax[0].grid(linestyle=':', alpha=0.7)
ax[1].grid(linestyle=':', alpha=0.7)


# Finish
ax[0].legend()
ax[1].legend()
plt.tight_layout()
fig.savefig('fom.png', dpi=300)




# %% Backyard

## full scatter
# for i, A in enumerate(['La', 'Pr', 'Nd']):
#     dfi = data[data[A]>=0.6]  
#     ax[0].scatter(dfi['TC'].values, dfi['ME'].values, s=15, c=colors[i], alpha=0.1, label=f'A site: {A}', zorder=1, marker='.')


# for i, A in enumerate(['La', 'Pr', 'Nd']):
#     dfi = data[data[A]>=0.6]  
#     ax[1].scatter(dfi['TC'].values, dfi['RCP'].values, s=15, c=colors[i], alpha=0.1, label=f'A site: {A}', zorder=1, marker='.')
    