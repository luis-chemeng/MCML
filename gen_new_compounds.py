#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Objetive: Systematic generation of potential new perovskites in the chemical space of the collected data

@author: research
"""

# %% Import Modeules

# Basic
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import json
import time

# Custom
import utilities as u



# %% Open files

dsa = pd.read_csv('dsa_TC.csv')


# %% Get list of site A elements and B site elements

A_sites, B_sites = [], []
A_sites_p, B_sites_p = [], []
for comp in list(dsa['Compound']):
    e, p = u.conv_c2in(comp, mode='Whitout O')
    A_p, B_p = u.split_list(p)
    A_sites.append(e[:len(A_p)])
    B_sites.append(e[len(A_p):])
    A_sites_p.append(A_p)
    B_sites_p.append(B_p)

unilist_sites_A = list(set([tuple(set(site_i)) for site_i in A_sites]))
unilist_sites_B = list(set([tuple(set(site_i)) for site_i in B_sites]))

elems_A = []
iter = 0
for tup in unilist_sites_A:
    
    if 'N' in list(tup):
        print(iter)
        print(tup)

    elems_A = elems_A + list(tup)
    iter += 1

elems_A = list(set(elems_A))


elems_B = []
for tup in unilist_sites_B:
    elems_B = elems_B + list(tup)

elems_B = list(set(elems_B))


# %% Systematic generation of new potential perovskites

sAs = ['La', 'Pr', 'Nd']
sBs = ['Mn']

cAs = [0.6]

cBs = [0.8]

comp_site_As = []

for sA in sAs:
    for cA in cAs:
        for i, sA1 in enumerate(elems_A):
            for sA2 in elems_A[i:]:
                for cA1 in np.arange(0.05, 1-cA, 0.05):
                    cA2 = 1-cA-cA1
                    siteA = sA+str(cA)+sA1+str(cA1)+sA2+str(cA2)
                    siteA  = u.collapse_formula(siteA)
                    comp_site_As.append(siteA)
                    if sA1 == sA2:
                        break

comp_site_As = list(set(comp_site_As))        


comp_site_Bs = []

for sB in sBs:
    for cB in cBs:
        for i, sB1 in enumerate(elems_B):
            for sB2 in elems_B[i:]:
                for cB1 in np.arange(0.05, 1-cB, 0.05):
                    cB2 = 1-cB-cB1
                    siteB = sB+str(cB)+sB1+str(cB1)+sB2+str(cB2)
                    siteB  = u.collapse_formula(siteB)
                    comp_site_Bs.append(siteB)
                    if sB1 == sB2:
                        break

comp_site_Bs = list(set(comp_site_Bs))

new_comps = []

for siteA in comp_site_As:
    for siteB in comp_site_Bs:
        new_comps.append(siteA+siteB+'O3')

new_comps = list(set(new_comps))

# %% Check all compositions are correct

for i in new_comps:
    t1 = u.conv_c2in(i, mode='Whitout O')[1]
    t2 = np.abs(np.array(t1).sum()-2)
    if t2>0.01:
        print(f'ERROR: {i} with value of {t2}.')

# %% Save

with open('new_comps.json', 'w') as file:
    json.dump(new_comps, file)

# %% Featurization mean type

stime = time.time()

lxi = []
for i, comp in enumerate(new_comps):
    xi = u.convert(comp, mode='Whitout O')
    lxi.append(xi)

X = np.concatenate(lxi, axis=0)

etime = time.time()

print(etime - stime)

rules = ['Sum', 'Mean', 'Var', 'Geo', 'Har', 'Min', 'Max']
col_names = [rules[j]+' '+u.prop_names[i] for j in range(len(rules)) for i in range(len(u.prop_names))]

df_X = pd.DataFrame(X, columns=col_names)

# %% Featurization tolerance factor like

stime = time.time()

lzi = []
for i, comp in enumerate(new_comps):
    tol = u.get_tol_combs(*u.conv_c2in(comp, mode='Whitout O'))
    lzi.append(tol.values)
    # if i==5:
    #     break
    
Z = np.concatenate(lzi, axis=0)

etime = time.time()

print(etime - stime)


df_Z = pd.DataFrame(Z, columns=tol.columns)
Z_nan_columns = pd.Index(['TolF_gs_bandgap', 
                          'TolF_gs_mag_moment',
                          'TolF_num_f_unfilled',
                          'TolF_num_f_valence',
                          'TolF_num_s_unfilled'])
df_Z.drop(columns=Z_nan_columns, inplace=True)

# %% Featurization MinDelta

stime = time.time()

lwi = []
for i, comp in enumerate(new_comps):
    delta = u.get_delta_combs(*u.conv_c2in(comp, mode='Whitout O'))
    lwi.append(delta.values)

W = np.concatenate(lwi, axis=0)

etime = time.time()

print(etime - stime)

df_W = pd.DataFrame(W, columns=delta.columns)
W_nan_columns = df_W.columns[df_W.isna().any()]
df_W.drop(columns=W_nan_columns, inplace=True)

# %% Save

df_T = pd.concat([df_X, df_Z, df_W], axis=1)

df_T.to_csv('df_new_comps_featurized.csv')

