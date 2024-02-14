#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Auxiliary functions

@author: research
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


# Import modules

import re

import warnings
warnings.filterwarnings('default')


prop2 = pd.read_excel('AtomicProps.xlsx')
prop2.rename(columns={'Unnamed: 0' : 'Symbol'}, inplace=True)
prop_names = prop2.columns.tolist()[1:]

d_prop = prop2.to_dict(orient='records')
dict_prop = dict()
for d in d_prop:
    l_values = []
    for prop_name in prop_names:
        l_values.append(d[prop_name])
    dict_prop.update({d['Symbol'] : l_values})

def h_mean(props, weights):
  mean = np.zeros(props.shape[1])
  for j in range(props.shape[1]):
    prop_colj = props[:,j]
    if np.all(prop_colj):
      mean_colj = weights.sum()/(weights.reshape(-1, 1)/prop_colj.reshape(-1, 1)).sum()
    else:
      mean_colj = 0
    mean[j] = mean_colj

  return mean


def conv_c2in(comp, mode='Whit O'):
    '''
    Coverting the compound string into the elements and their proportions.
    '''
    elems = re.findall('[A-Z][a-z]?', comp)
    weights = re.split('[A-Z][a-z]?', comp)[1:]
    float_weights = []
    for w in weights:
        if w=='':
            float_weights.append(1)
        else:
            float_weights.append(float(w))
            
    # float_weights = np.array(float_weights)

    if mode=='Whit O':
        return elems, float_weights
    elif mode=='Whitout O':
        return elems[:-1], float_weights[:-1]
    else:
        print('Invalid mode parameter.')
        return None
    

def combine_atom_prop(elems, weights):
    '''
    Combine the atomic propeties their proportions in the seven ways proposed 
    by Callun.
    '''
    props = [dict_prop[elem] for elem in elems]
    props = np.array(props)
    
    w = np.array(weights).reshape(-1, 1)
    wtot = w.sum()
    
    # Weighted sum, average and variance
    w_mat = w*props
    w_sum = w_mat.sum(axis=0)
    w_avg = w_sum/wtot
    w_var = (w*np.power(props - w_avg, 2)).sum(axis=0)/wtot
    
    # Geometric average
    w_mat = np.abs(np.power(props.astype(np.cdouble), w/wtot))
    w_prod = w_mat.prod(axis=0)
    
    # Harmonic average
    w_har = h_mean(props, w)
    
    # Maximun and Minimun pooling
    p_min = np.min(props, axis=0)
    p_max = np.max(props, axis=0)
    
    return np.array([w_sum, w_avg, w_var, w_prod, w_har, p_min, p_max]).reshape(1, -1)

def convert(x, mode='With O'):
    return combine_atom_prop(*conv_c2in(x, mode=mode))

def split_list(input_list):

    lenght = len(input_list)
    
    arr = np.array(input_list)
    
    best_score = 10
    for i in range(lenght-1):
        arr1 = arr[:(i+1)]
        arr2 = arr[(i+1):]
        score = np.abs(arr1.sum()-1) + np.abs(arr2.sum()-1)
        if score < best_score:
            best_score = score
            list1 = list(arr1)
            list2 = list(arr2)
            
    return list1, list2

def get_tol_combs(elems, weights):
    '''
    Combine the atomic propeties in new ways
    '''
    props = [dict_prop[elem] for elem in elems]
    props = np.array(props)
    
    # Fist identify site A and site B
    ws_site_A, ws_site_B = split_list(weights)
    
    props_site_A = (np.array(ws_site_A).reshape(-1, 1)*props[:len(ws_site_A),:]).sum(axis=0)
    props_site_B = (np.array(ws_site_B).reshape(-1, 1)*props[len(ws_site_A):,:]).sum(axis=0)
    props_site_O = np.array(dict_prop['O']).reshape(1, -1)
    
    tol = (props_site_A + props_site_O)/(props_site_B + props_site_O)
    return pd.DataFrame(tol, columns=['TolF_'+pn for pn in prop_names])

def get_delta_combs(elems, weights):
    '''
    Combine the atomic propeties in new ways
    '''
    props = [dict_prop[elem] for elem in elems]
    props = np.array(props)
    
    # Fist identify site A and site B
    ws_site_A, ws_site_B = split_list(weights)
    
    props_site_A = (np.array(ws_site_A).reshape(-1, 1)*props[:len(ws_site_A),:]).sum(axis=0).reshape(1, -1)
    props_site_B = (np.array(ws_site_B).reshape(-1, 1)*props[len(ws_site_A):,:]).sum(axis=0).reshape(1, -1)
    props_site_O = np.array(dict_prop['O']).reshape(1, -1)
    
    delta = np.concatenate((props_site_A - props_site_O, props_site_B - props_site_O), axis=0).min(axis=0).reshape(1, -1)
    
    return pd.DataFrame(delta, columns=['MinDelta_'+pn for pn in prop_names])
    


# Funtion to create parity plot
def parity_plot(model, X_train, X_test, y_train, y_test):
  fig, ax = plt.subplots(figsize=(8, 8))

  y_pred_train = model.predict(X_train)
  y_pred_test = model.predict(X_test)

  ax.scatter(y_train, y_pred_train, alpha=0.8)
  ax.scatter(y_test, y_pred_test, alpha=0.8)
  ax.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='--')
  ax.text(0.8, 0.05, s='R2 Test: {:.3f}'.format(r2_score(y_test, y_pred_test)), fontsize=12, transform=ax.transAxes)
  ax.text(0.8, 0.1, s='R2 Train: {:.3f}'.format(r2_score(y_train, y_pred_train)), fontsize=12, transform=ax.transAxes)
  ax.text(0.8, 0.15, s='MAE Test: {:.1f}'.format(mean_absolute_error(y_test, y_pred_test)), fontsize=12, transform=ax.transAxes)
  ax.text(0.8, 0.2, s='MAE Train: {:.1f}'.format(mean_absolute_error(y_train, y_pred_train)), fontsize=12, transform=ax.transAxes)
  ax.text(0.8, 0.25, s='RMSE Test: {:.1f}'.format(np.sqrt(mean_squared_error(y_test, y_pred_test))), fontsize=12, transform=ax.transAxes)
  ax.text(0.8, 0.3, s='RMSE Train: {:.1f}'.format(np.sqrt(mean_squared_error(y_train, y_pred_train))), fontsize=12, transform=ax.transAxes)

  ax.set_title('Parity Plot')
  ax.set_xlabel('Actual Values')
  ax.set_ylabel('Predicted Values')
  
  
def find_lucky_seed(df, mod=6):
    for seed in range(5000):
      t1, t2 = train_test_split(df, test_size=0.2, random_state=seed)
      li1 = []

      cols = df.columns[mod:]
      

      for col in cols:
        con1 = t1[col][t1[col] > 0].count()
        con2 = t2[col][t2[col] > 0].count()
        li1.append(con1/(con1+con2))

      if np.all(np.array(li1)>0.5):
        print(f'The lucky seed is: {seed}')
        break
    
    return seed
    
    

def collapse_formula(raw_formula):
    # Use regular expression to find all occurrences of letter patterns followed by a number
    matches = re.findall(r'([A-Za-z]+)(\d+(\.\d+)?)', raw_formula)

    # Create a dictionary to store the sum for each letter pattern
    letter_sums = {}

    for match in matches:
        letter_pattern, value = match[0], float(match[1])
        letter_sums[letter_pattern] = letter_sums.get(letter_pattern, 0) + value

    # Round the sum for each letter pattern to the nearest integer
    sums = {letter: round(value, 2) for letter, value in letter_sums.items()}

    # Create the desired output string
    formula = ''.join(f'{letter}{value}' for letter, value in sums.items())

    return formula    


def format_str_to_latex(input_string):
    # Define a regular expression pattern to match chemical elements followed by decimal numbers or integers
    pattern = re.compile(r'([A-Z][a-z]*)([0-9]*\.?[0-9]+)')

    # Use re.sub to replace matched patterns with the desired format
    formatted_string = pattern.sub(r'\1_{\2}', input_string)

    formatted_string = '$' + formatted_string + '$'

    return formatted_string

# Funtion to create parity plot
def parity_plot_(ax, y_pred_train, y_pred_test, y_train, y_test, fs=10, xt=0.65, error_lines=[0.2, 0.1, -0.1, -0.2]):

    iden = np.arange(min(y_train), max(y_train))

    for error_line in error_lines:
        if error_line > 0:
            ax.plot(iden, (1+error_line)*iden, linestyle='--', color='gray', alpha=0.8, label=str(error_line*100)+'% AAD')
        else:
            ax.plot(iden, (1+error_line)*iden, linestyle='--', color='gray', alpha=0.8,)
    ax.scatter(y_train, y_pred_train, alpha=0.8, label='Train')
    ax.scatter(y_test, y_pred_test, alpha=0.8, label='Test')
    ax.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='--')
    ax.text(xt, 0.05, s=r'$R^{2}_{Test}=$'+' {:.3f}'.format(r2_score(y_test, y_pred_test)), fontsize=fs, transform=ax.transAxes)
    ax.text(xt, 0.1, s=r'$R^{2}_{Train}=$'+' {:.3f}'.format(r2_score(y_train, y_pred_train)), fontsize=fs, transform=ax.transAxes)
    ax.text(xt, 0.15, s=r'$MAE_{Test}=$'+' {:.1f}'.format(mean_absolute_error(y_test, y_pred_test)), fontsize=fs, transform=ax.transAxes)
    ax.text(xt, 0.2, s=r'$MAE_{Train}=$'+' {:.1f}'.format(mean_absolute_error(y_train, y_pred_train)), fontsize=fs, transform=ax.transAxes)
    ax.text(xt, 0.25, s=r'$RMSE_{Test}=$'+' {:.1f}'.format(np.sqrt(mean_squared_error(y_test, y_pred_test))), fontsize=fs, transform=ax.transAxes)
    ax.text(xt, 0.3, s=r'$RMSE_{Train}=$'+' {:.1f}'.format(np.sqrt(mean_squared_error(y_train, y_pred_train))), fontsize=fs, transform=ax.transAxes)

    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    
    return ax

def histogram_residuals_(ax, y_pred_train, y_pred_test, y_train, y_test, bins=15):
    
    resi_train = (y_pred_train - y_train)#/y_train
    resi_test = (y_pred_test - y_test)#/y_test
    
    resi = np.concatenate([resi_train, resi_test])
    bins = np.histogram_bin_edges(resi, bins='scott')
    
    ax.hist(resi_train, bins=bins, zorder=98, edgecolor='black', linewidth=0.8, density=True, alpha=0.8)
    ax.hist(resi_test, bins=bins, zorder=99, edgecolor='black', linewidth=0.8, density=True, alpha=0.8)
    
    
    return ax
    
    
