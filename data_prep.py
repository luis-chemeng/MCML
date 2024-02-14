#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Objetive: Prepare the data for machine learning models

@author: research
"""

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


# Dictinonary for relate abbreviation and proeprty
d_prop = {
    'TC' : 'Curie Temperature',
    'ME' : 'Maximum Magnetic Entropy',
    'RCP' : 'RCP',
}

# Column names
rules = ['Sum', 'Mean', 'Var', 'Geo', 'Har', 'Min', 'Max']
col_names = [rules[j]+' '+u.prop_names[i] for j in range(len(rules)) for i in range(len(u.prop_names))]

# Load dataset
dataset = pd.read_excel('./DB_mc.xlsx', sheet_name='2')

prop = 'RCP'

# %% DatSets Creation 

# Filtrar por Temperatura de Curie, compociçao e valor
if prop=='TC':
  ds = dataset[dataset['Property']=='Tc (K)'][['Compound','D_crystalite [nm]', 'Magnetic Field','Value', 'Reference']]
elif prop=='ME':
  ds = dataset[dataset['Property']=='ME (J/kg.K)'][['Compound','D_crystalite [nm]', 'Magnetic Field', 'Value', 'Reference']]
elif prop=='RCP':
  ds = dataset[dataset['Property']=='RCP (J/Kg)'][['Compound','D_crystalite [nm]', 'Magnetic Field', 'Value', 'Reference']]

## Aply filters
# Left only stechemometric perovskites
ds = ds[(ds['Compound'].str[-2:] == 'O6')|(ds['Compound'].str[-2:]=='O3')]
ds = ds[ds['Compound'].str[-2:]=='O3']

# Apagar registros que tem parêntese e commas
ds.drop(ds[ds['Compound'].str.contains('\(')].index, inplace=True)
print('Tamanho de dataframe depois de excluir as barras (\\) é: {} filas.'.format(ds.shape))
ds.drop(ds[ds['Compound'].str.contains(',')].index, inplace=True)
print('Tamanho de dataframe depois de excluir as commas (,) é: {} filas.'.format(ds.shape))
ds.drop(ds[ds['Compound'].str.contains('-')].index, inplace=True)
print('Tamanho de dataframe depois de excluir os lines (-) é: {} filas.'.format(ds.shape))

# Remove the data without reference
ds.drop(ds[ds['Reference']=='-'].index, inplace=True)
print('Tamanho de dataframe depois de excluir os registros sem referencia é: {} filas.'.format(ds.shape))

# Remove some outliers
ds.drop(ds[ds['Reference']=='10.1016/j.jallcom.2022.164583'].index, inplace=True)

# Resetear index
ds.reset_index(inplace=True)

ds.info()

dsa = ds.copy()

if prop=='ME' or prop=='RCP':
    dsa['CompoundpH'] = dsa['Compound'] + '_' + dsa['Magnetic Field'].astype(str)
    median_values = dsa.groupby('CompoundpH')['Value'].median().reset_index()
    dsa = pd.merge(dsa, median_values, on=['CompoundpH'], how='inner')
    dsa.drop_duplicates(subset=['CompoundpH', 'Value_y'], inplace=True)
else:    
    median_values = dsa.groupby('Compound')['Value'].median().reset_index()
    dsa = pd.merge(dsa, median_values, on=['Compound'], how='inner')
    dsa.drop_duplicates(subset=['Compound', 'Value_y'], inplace=True)

dsa.drop(columns=['Value_x'], inplace=True)
dsa.rename(columns={'Value_y': 'Value', 'index' : 'id'}, inplace=True)
dsa.reset_index(inplace=True)
dsa.drop(columns=['index'], inplace=True)


dsb = ds.copy()
dsb.dropna(subset=['D_crystalite [nm]'], inplace=True)
dsb.rename(columns={'index' : 'id'}, inplace=True)
dsb.reset_index(inplace=True)
dsb.drop(columns=['index'], inplace=True)

# %% Element concentrarion in DSA

elements = []

list_procesed_comps = []

for i in range(dsa.shape[0]):
  comp = dsa.iloc[i].Compound
  e, p = u.conv_c2in(comp, mode='Whit O')
  elements = elements + e
  list_procesed_comps.append({key: value for key, value in zip(e, p)})

elements = list(set(elements))

dfa = pd.DataFrame(list_procesed_comps).fillna(0)

dsa = dsa.merge(dfa, left_index=True, right_index=True)

dsa.to_csv(f'dsa_{prop}.csv', index=False)

# %% Element concentrarion in DSB

elements = []

list_procesed_comps = []

for i in range(dsb.shape[0]):
  comp = dsb.iloc[i].Compound
  e, p = u.conv_c2in(comp, mode='Whit O')
  elements = elements + e
  list_procesed_comps.append({key: value for key, value in zip(e, p)})

elements = list(set(elements))

dfb = pd.DataFrame(list_procesed_comps).fillna(0)

dsb = dsb.merge(dfb, left_index=True, right_index=True)

dsb.to_csv(f'dsb_{prop}.csv', index=False)

# %% Featurization DSA

"""-DSA-"""

# Criar dados de entrada para a rede neuronal
for i in range(dsa.shape[0]):
    xi = u.convert(dsa.iloc[i].Compound, mode='Whitout O')
    xi = np.concatenate((xi, np.array([dsa['Magnetic Field'].iloc[i]]).reshape(-1, 1)), axis=1)
    yi = np.array([dsa.iloc[i].Value]).reshape(-1, 1)

    if i==0:
        X = xi
        Y = yi
    else:
        X = np.concatenate((X, xi), axis=0)
        Y = np.concatenate((Y, yi), axis=0)

V = np.concatenate((X, Y), axis=1)

df_V = pd.DataFrame(V, columns=col_names+['H', prop])

X_ = df_V.dropna().iloc[:,:-1].values
Y_ = df_V.dropna().iloc[:,-1].values.reshape(-1, 1)

# Features named Tolerance-like
for i in range(dsa.shape[0]):
  tol = u.get_tol_combs(*u.conv_c2in(dsa.iloc[i].Compound, mode='Whitout O'))
  zi = tol.values
  if i==0:
    Z = zi
  else:
    Z = np.concatenate((Z, zi), axis=0)

df_Z = pd.DataFrame(Z, columns=tol.columns)

Z_nan_columns = df_Z.columns[df_Z.isna().any()]

df_Z.drop(columns=Z_nan_columns, inplace=True)

# Features named MinDelta
for i in range(dsa.shape[0]):
  delta = u.get_delta_combs(*u.conv_c2in(dsa.iloc[i].Compound, mode='Whitout O'))
  wi = delta.values
  if i==0:
    W = wi
  else:
    W = np.concatenate((W, wi), axis=0)

df_W = pd.DataFrame(W, columns=delta.columns)

W_nan_columns = df_W.columns[df_W.isna().any()]

df_W.drop(columns=W_nan_columns, inplace=True)

# Join all the features
df_T = pd.concat([df_V, df_Z, df_W], axis=1)

df_T = df_T[df_V.columns.to_list()[:-2]+df_Z.columns.to_list()+df_W.columns.to_list()+['H', prop]]

df_T.to_csv(f'feats_dsa_{prop}.csv', index=False)

# %% Featurization DSB

"""-DSB-"""

# Criar dados de entrada para a rede neuronal
for i in range(dsb.shape[0]):
    xi = u.convert(dsb.iloc[i].Compound, mode='Whitout O')
    xi = np.concatenate((xi, np.array([dsb['D_crystalite [nm]'].iloc[i]]).reshape(-1, 1)), axis=1)
    xi = np.concatenate((xi, np.array([dsb['Magnetic Field'].iloc[i]]).reshape(-1, 1)), axis=1)
    yi = np.array([dsb.iloc[i].Value]).reshape(-1, 1)

    if i==0:
        X = xi
        Y = yi
    else:
        X = np.concatenate((X, xi), axis=0)
        Y = np.concatenate((Y, yi), axis=0)

V = np.concatenate((X, Y), axis=1)

df_V = pd.DataFrame(V, columns=col_names+['D', 'H', prop])

X_ = df_V.dropna().iloc[:,:-1].values
Y_ = df_V.dropna().iloc[:,-1].values.reshape(-1, 1)

# Features named Tolerance-like
for i in range(dsb.shape[0]):
  tol = u.get_tol_combs(*u.conv_c2in(dsb.iloc[i].Compound, mode='Whitout O'))
  zi = tol.values
  if i==0:
    Z = zi
  else:
    Z = np.concatenate((Z, zi), axis=0)

df_Z = pd.DataFrame(Z, columns=tol.columns)

Z_nan_columns = df_Z.columns[df_Z.isna().any()]

df_Z.drop(columns=Z_nan_columns, inplace=True)

# Features named MinDelta
for i in range(dsb.shape[0]):
  delta = u.get_delta_combs(*u.conv_c2in(dsb.iloc[i].Compound, mode='Whitout O'))
  wi = delta.values
  if i==0:
    W = wi
  else:
    W = np.concatenate((W, wi), axis=0)

df_W = pd.DataFrame(W, columns=delta.columns)

W_nan_columns = df_W.columns[df_W.isna().any()]

df_W.drop(columns=W_nan_columns, inplace=True)

# Join all the features
df_T = pd.concat([df_V, df_Z, df_W], axis=1)

df_T = df_T[df_V.columns.to_list()[:-3]+df_Z.columns.to_list()+df_W.columns.to_list()+['D', 'H', prop]]

df_T.to_csv(f'feats_dsb_{prop}.csv', index=False)


