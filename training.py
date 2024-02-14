#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Objetive: Optimize hyper parameters and get valdiation and testing scores

@author: research
"""

# %% Modules

# Basic operation and graph modules
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import dcor

# Auxiliary modules
import utilities as u
import os
import time
import shutil

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

prop = 'RCP'

wd = 'RCP4'  # Working directory

dsa = pd.read_csv(f'dsa_{prop}.csv')
dsb = pd.read_csv(f'dsb_{prop}.csv')
dfa = pd.read_csv(f'feats_dsa_{prop}.csv')
dfb = pd.read_csv(f'feats_dsb_{prop}.csv')

comp_feats = dfa.columns[:517].to_list()

with open(f'{wd}/RFE.json', 'r') as file:
    dict_rfe_best = json.load(file)

with open(f'{wd}/COR.json', 'r') as file:
    dict_cor_best = json.load(file)

dict_group_feats = {
    'All517': comp_feats[:517],
    'Sum': comp_feats[:58],
    'Mean': comp_feats[58:58*2],
    'Var': comp_feats[58*2:58*3],
    'Geo': comp_feats[58*3:58*4],
    'Har': comp_feats[58*4:58*5],
    'Min': comp_feats[58*5:58*6],
    'Max': comp_feats[58*6:58*7],
    'TolF': comp_feats[58*7:459],
    'MinDelta': comp_feats[459:517],
}


n_feats = ['10',]


for n_feat in n_feats:
    dict_group_feats.update(
        {'SC'+n_feat: dict_cor_best['SC_decorr'][:int(n_feat)]})
    dict_group_feats.update(
        {'PC'+n_feat: dict_cor_best['PC_decorr'][:int(n_feat)]})
    dict_group_feats.update(
        {'DC'+n_feat: dict_cor_best['DC_decorr'][:int(n_feat)]})
    dict_group_feats.update({'RFE'+n_feat: dict_rfe_best[n_feat]})


# %% Training

istest = False
train_with_hp = True # To do bayesian optimization with ANN instead of grid search



# Artificial Neural Network
ANN = MLPRegressor(max_iter=100000, tol=0.0001, learning_rate_init=0.05,
                   solver="adam", learning_rate="adaptive", verbose=0, random_state=1,)

ANN_hl = [(4,), (8,), (12,), (4, 4), (12, 4), (12, 12), (4, 4, 4)]
ANN_act = ['relu', 'logistic']
ANN_al = [0.001, 0.01, 0.1, 1, 10]

ANN_param_grid = {
    'hidden_layer_sizes': ANN_hl,
    'activation': ANN_act,
    'alpha': ANN_al,
}


space = {
    'hidden_layer_sizes': hp.choice('hidden_layer_sizes', ANN_hl),
    'alpha': hp.choice('alpha', ANN_al),
    'activation': hp.choice('activation', ANN_act),
}

# Define the objective function for hyperparameter optimization (hyperopt)
def objective(params):
    model = MLPRegressor(hidden_layer_sizes=params['hidden_layer_sizes'],
                         alpha=params['alpha'],
                         activation=params['activation'],
                         random_state=1,
                         max_iter=100000,
                         tol=0.0001,
                         learning_rate_init=0.05,
                         solver="adam",
                         learning_rate="adaptive",
                         verbose=0,)

    # Use cross-validation to evaluate the performance
    score = -np.mean(cross_val_score(model, X_train, Y_train,
                     cv=5, scoring='neg_mean_squared_error'))

    return score


# XGBoost Regressor
XGB = XGBRegressor(random_state=1)

XGB_param_grid = {
    'n_estimators': [25, 100, 400],
    'max_depth': [2, 5, 10, None],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
}


# Random forest regressor
RF = RandomForestRegressor(random_state=1)

RF_param_grid = {
    'n_estimators': [25, 100, 400],
    'max_depth': [2, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 3, 5]
}


# Kernel ridge regression
KRR = KernelRidge()

KRR_param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1, 10],  # Regularization strength
    'kernel': ['rbf', 'poly'],  # Kernel type
    'gamma': [0.001, 0.01, 0.1, 1, 10]  # Kernel coefficient for 'rbf' and 'poly'
}

data2 = []

start_time = time.time()

file_path = f'{wd}/res.xlsx'

if os.path.exists(file_path):
    # Creating a backup by appending a timestamp to the original file name
    backup_path = file_path.replace(
        '.xlsx', f'_backup_{int(time.time())}.xlsx')

    # Copying the contents of the original file to the backup file
    with open(file_path, 'rb') as original_file, open(backup_path, 'wb') as backup_file:
        shutil.copyfileobj(original_file, backup_file)

    print(f'Backup created: {backup_path}')
else:
    print(f'The file {file_path} does not exist.')

mlms = [KRR, ANN, XGB, RF]
mlm_names = ['RR', 'ANN', 'XGB', 'RF']
param_grids = [KRR_param_grid, ANN_param_grid, XGB_param_grid, RF_param_grid]

# for k in range(1,2):
for k in [0]:

    mlm = mlms[k]
    mlm_name = mlm_names[k]
    param_grid = param_grids[k]

    for key, value in dict_group_feats.items():

        n_feat = key
        pred = value

        for dtype in ['A', 'B']:

            if dtype == 'A':
                df = dfa
                lucky_seed = u.find_lucky_seed(dsa)
                lucky_seed = 42
                if prop == 'TC':
                    X_sub = df[pred]
                else:
                    X_sub = df[pred+['H']]
            else:
                df = dfb
                lucky_seed = u.find_lucky_seed(dsb)
                lucky_seed = 42
                if prop == 'TC':
                    X_sub = df[pred+['D']]
                else:
                    X_sub = df[pred+['H', 'D']]

            Y_sub = np.array(df[prop])

            scaler = StandardScaler()
            X_sub_scaled = scaler.fit_transform(X_sub)

            X_train, X_test, Y_train, Y_test = train_test_split(
                X_sub_scaled, Y_sub, test_size=0.2, random_state=lucky_seed)

            code = mlm_name+'_'+dtype+'_'+str(n_feat)
            iswasdid = False  # code in codes

            mlmsavename = wd+'/'+mlm_name+'_'+dtype+'_'+str(n_feat)+'.joblib'

            if not istest and not iswasdid:

                print(mlmsavename)

                if mlm_name == 'ANN' and train_with_hp:
                    trials = Trials()
                    best = fmin(fn=objective, space=space,
                                max_evals=45, algo=tpe.suggest)

                    best_model = MLPRegressor(hidden_layer_sizes=ANN_hl[best['hidden_layer_sizes']],
                                              alpha=ANN_al[best['alpha']],
                                              activation=ANN_act[best['activation']],
                                              random_state=1,
                                              max_iter=100000,
                                              tol=0.0001,
                                              learning_rate_init=0.05,
                                              solver="adam",
                                              learning_rate="adaptive",
                                              verbose=0,)

                    best_model.fit(X_train, Y_train)

                else:

                    grid_search = GridSearchCV(
                        mlm, param_grid, scoring='neg_mean_squared_error', cv=5)
                    grid_search.fit(X_train, Y_train, )

                    best_model = grid_search.best_estimator_

                Y_train_pred = best_model.predict(X_train)
                Y_test_pred = best_model.predict(X_test)
                mse_test = mean_squared_error(Y_test, Y_test_pred)
                mae_test = mean_absolute_error(Y_test, Y_test_pred)
                r2_test = r2_score(Y_test, Y_test_pred)
                mse_train = mean_squared_error(Y_train, Y_train_pred)
                mae_train = mean_absolute_error(Y_train, Y_train_pred)
                r2_train = r2_score(Y_train, Y_train_pred)

                data2.append([n_feat, dtype, mlm_name, r2_test, np.sqrt(
                    mse_test), mae_test, r2_train, np.sqrt(mse_train), mae_train])

                joblib.dump(best_model, mlmsavename)

                print(time.time()-start_time)

        columns = ['N_feat', 'Dataset', 'ML method', 'r2_test',
                   'RMSE_test', 'MAE_test', 'r2_train', 'RMSE_train', 'MAE_train']
        df2 = pd.DataFrame(data2, columns=columns)

        df2.to_excel(f'{wd}/res.xlsx')

end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print(f"Execution time: {elapsed_time} seconds")

with open(f'{wd}/log.txt', 'w') as file:
    file.write(f"Elapsed Time: {elapsed_time}\n")
