# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 15:37:22 2022

@author: mz52
"""

from sklearn.datasets import load_boston
import numpy as np
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
from itertools import combinations
import time
import asgl

#%% Preprocessing

N_iteration = 10

#Loading datasets
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    boston_dataset = load_boston()
X = boston_dataset.data
y = boston_dataset.target
feature_names = boston_dataset.feature_names
print("** Preprocessing **\n")
print("%-25s%-25s"%('Dataset:','Boston House Prices'))
print("%-24s"%('Dataset dimensions:'), X.shape)
print("%-24s"%('Feature names:'), feature_names.tolist())

def splitter_normalizer(X, y):
    # Splitting
    test_size = 0.2   # Test portion of the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=None)      # Test size = 20%
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=None) # Validation size = 0.25 x 0.8 = 0.2 (20%)
    # Standardization
    std_scaler = StandardScaler()
    X_train = std_scaler.fit_transform(X_train)
    X_test = std_scaler.transform(X_test)
    X_val = std_scaler.transform(X_val)
    X_train_val = np.append(X_train, X_val, axis=0)
    y_train_val = np.append(y_train, y_val, axis=0)
    return X_train, X_val, X_train_val, X_test, y_train, y_val, y_train_val, y_test

#%% OLS
tic = time.time()
ls_mse_vec = np.zeros(N_iteration)
ls_r2_vec = np.zeros(N_iteration)
ls_best_mse_it = np.inf
for it in range(N_iteration):
    X_train, X_val, X_train_val, X_test, y_train, y_val, y_train_val, y_test = splitter_normalizer(X,y)
    # Least Square
    ls_regr = linear_model.LinearRegression()
    ls_model = ls_regr.fit(X_train_val, y_train_val)
    ls_pred_y_test = ls_model.predict(X_test)
    ls_test_mse = mean_squared_error(y_test, ls_pred_y_test)
    ls_test_r2 = r2_score(y_test, ls_pred_y_test)
    ls_mse_vec[it] = ls_test_mse
    ls_r2_vec[it] = ls_test_r2
    if ls_test_mse < ls_best_mse_it:
        ls_best_coefs = ls_model.coef_
toc = time.time()
ls_time = toc - tic

print ('\n** Ordinary Least Square **')
print('===================================')
print ("%-15s%-15s"%('MSE','R2'))
print('-----------------------------------')
print("%-15f%-15f"%(np.mean(ls_mse_vec), np.mean(ls_r2_vec)))
print('-----------------------------------')
print("Minimum Test MSE")
print("Processed in %.6f seconds"%ls_time)
print('===================================')


#%% Ridge
tic = time.time()
ridge_mse_vec = np.zeros(N_iteration)
ridge_r2_vec = np.zeros(N_iteration)
ridge_best_mse_it = np.inf
for it in range(N_iteration):
    X_train, X_val, X_train_val, X_test, y_train, y_val, y_train_val, y_test = splitter_normalizer(X,y)
    n_alphas = 100
    alpha_vec = 10 ** np.linspace(-1,3,n_alphas)
    ridge_best_alpha = -1
    ridge_best_mse = np.inf
    for i in range(n_alphas):
        alpha = alpha_vec[i]
        ridge_regr = linear_model.Ridge(alpha=alpha)
        ridge_model = ridge_regr.fit(X_train, y_train)
        ridge_pred_y_val = ridge_model.predict(X_val)
        ridge_val_mse = mean_squared_error(y_val, ridge_pred_y_val)
        if ridge_val_mse < ridge_best_mse:
            ridge_best_mse = ridge_val_mse
            ridge_best_alpha = alpha
    ridge_regr = linear_model.Ridge(alpha=ridge_best_alpha)
    ridge_model = ridge_regr.fit(X_train_val, y_train_val)
    ridge_pred_y_test = ridge_model.predict(X_test)
    ridge_test_mse = mean_squared_error(y_test, ridge_pred_y_test)
    ridge_test_r2 = r2_score(y_test, ridge_pred_y_test)
    ridge_mse_vec[it] = ridge_test_mse
    ridge_r2_vec[it] = ridge_test_r2
    if ridge_test_mse < ridge_best_mse_it:
        ridge_best_coefs = ridge_model.coef_
        ridge_min_mse_alpha = ridge_best_alpha
toc = time.time()
ridge_time = toc - tic

print ('\n** Ridge Regularization **')
print('===================================')
print ("%-10s%-15s%-15s"%('\u03B1','MSE','R2'))
print('-----------------------------------')
print("%-10.3f%-15f%-15f"%(ridge_min_mse_alpha,np.mean(ridge_mse_vec), np.mean(ridge_r2_vec)))
print('-----------------------------------')
print("Minimum Test MSE")
print("Processed in %.6f seconds"%ridge_time)
print('===================================')


#%% Best Subset
tic = time.time()
bs_best_models = {}
for it in range(N_iteration):
    X_train, X_val, X_train_val, X_test, y_train, y_val, y_train_val, y_test = splitter_normalizer(X,y)
    feature_idxs = np.arange(X.shape[1])
    for i in range(1, X.shape[1]+1):
        bs_best_mse = np.inf
        idx = it*X.shape[1] + i
        for comb in combinations(feature_idxs, i):
            selected_features = np.asarray(comb)
            new_X_train = sm.add_constant(X_train[:,selected_features])
            new_X_val = sm.add_constant(X_val[:,selected_features])
            new_X_test = sm.add_constant(X_test[:,selected_features])
            est = sm.OLS(y_train,new_X_train)
            ols_est = est.fit()
            bs_pred_y_val = ols_est.predict(new_X_val)
            bs_val_mse = mean_squared_error(y_val,bs_pred_y_val)
            if bs_val_mse < bs_best_mse:
                bs_best_mse = bs_val_mse
                bs_pred_y_test = ols_est.predict(new_X_test)
                bs_test_mse = mean_squared_error(y_test,bs_pred_y_test)
                bs_test_r2 = r2_score(y_test, bs_pred_y_test)
                bs_best_models[idx] = selected_features, bs_test_mse, bs_test_r2, ols_est.params[1:]
toc = time.time()
bs_time = toc - tic
best_keys = np.zeros(X.shape[1])
for i in range(0,X.shape[1]):
    best_mse = np.inf
    for key, value in bs_best_models.items():
        if (key % X.shape[1] == i):
            if value[1] < best_mse:
                best_mse = value[1]
                best_key = key
    if i == 0:
        best_keys[X.shape[1]-1] = best_key
    else:
        best_keys[i-1] = best_key
print ('\n** Best Feature Selection **')
print('============================================================')
print ("%-5s%-15s%-15s%-15s"%('K','MSE','R2','Subset'))
print('------------------------------------------------------------')
for i in range(X.shape[1]):
    print("%-5i%-15f%-14f"%(i+1, bs_best_models[best_keys[i]][1], bs_best_models[best_keys[i]][2]), feature_names[bs_best_models[best_keys[i]][0]].tolist())
    
print('------------------------------------------------------------')
print("Selection Algorithm: Exhaustive")
print("Critirion: Minimizing MSE")
print("Processed in %.6f seconds"%bs_time)
print('============================================================')


#%% Forward Selection
tic = time.time()
fws_best_models = {}
for it in range(N_iteration):
    X_train, X_val, X_train_val, X_test, y_train, y_val, y_train_val, y_test = splitter_normalizer(X,y)
    remaining_features = feature_idxs
    selected_features = np.array([],dtype=int)
    for i in range(1, X.shape[1]+1):
        fws_best_mse = np.inf
        idx = it*X.shape[1] + i
        for comb in combinations(remaining_features, 1):
            temp_feature = np.asarray(comb)
            temp_subset = np.append(selected_features,temp_feature)
            new_X_train = sm.add_constant(X_train[:,temp_subset])
            new_X_val = sm.add_constant(X_val[:,temp_subset])
            est = sm.OLS(y_train,new_X_train)
            ols_est = est.fit()
            fws_pred_y_val = ols_est.predict(new_X_val)
            fws_val_mse = mean_squared_error(y_val,fws_pred_y_val)
            if fws_val_mse < fws_best_mse:
                fws_best_mse = fws_val_mse
                best_feature = temp_feature
        selected_features = np.append(selected_features,best_feature)
        new_X_train = sm.add_constant(X_train[:,selected_features])
        new_X_test = sm.add_constant(X_test[:,selected_features])
        est = sm.OLS(y_train,new_X_train)
        ols_est = est.fit()
        fws_pred_y_test = ols_est.predict(new_X_test)
        fws_test_mse = mean_squared_error(y_test,fws_pred_y_test)
        fws_test_r2 = r2_score(y_test, fws_pred_y_test)
        fws_best_models[idx] = selected_features, fws_test_mse, fws_test_r2, ols_est.params[1:], best_feature+1
        remaining_features = np.delete(remaining_features, np.where(remaining_features==best_feature)[0][0])
toc = time.time()
fws_time = toc - tic
best_keys = np.zeros(X.shape[1])
for i in range(0,X.shape[1]):
    best_mse = np.inf
    for key, value in fws_best_models.items():
        if (key % X.shape[1] == i):
            if value[1] < best_mse:
                best_mse = value[1]
                best_key = key
    if i == 0:
        best_keys[X.shape[1]-1] = best_key
    else:
        best_keys[i-1] = best_key
print ('\n** Forward Selection **')
print('============================================================')
print ("%-5s%-15s%-15s%-15s"%('K','MSE','R2','Subset'))
print('------------------------------------------------------------')
for i in range(X.shape[1]):
    print("%-5i%-15f%-14f"%(i+1, fws_best_models[best_keys[i]][1], fws_best_models[best_keys[i]][2]), feature_names[fws_best_models[best_keys[i]][0]].tolist())
    
print('------------------------------------------------------------')
print("Selection Algorithm: Forward Stepwise Selection")
print("Critirion: Minimizing MSE")
print("Processed in %.6f seconds"%fws_time)
print('============================================================')


#%% Backward Selection
tic = time.time()
bws_best_models = {}
for it in range(N_iteration):
    X_train, X_val, X_train_val, X_test, y_train, y_val, y_train_val, y_test = splitter_normalizer(X,y)
    remaining_features = feature_idxs
    for i in range(1, X.shape[1]+1):
        bws_best_mse = np.inf
        idx = it*X.shape[1] + i
        for comb in combinations(remaining_features, 1):
            temp_feature = np.asarray(comb)
            temp_subset = np.delete(remaining_features,np.where(remaining_features==temp_feature)[0][0])
            new_X_train = sm.add_constant(X_train[:,temp_subset])
            new_X_val = sm.add_constant(X_val[:,temp_subset])
            est = sm.OLS(y_train,new_X_train)
            ols_est = est.fit()
            bws_pred_y_val = ols_est.predict(new_X_val)
            bws_val_mse = mean_squared_error(y_val,bws_pred_y_val)
            if bws_val_mse < bws_best_mse:
                bws_best_mse = bws_val_mse
                worst_feature = temp_feature
        remaining_features = np.delete(remaining_features, np.where(remaining_features==worst_feature)[0][0])
        new_X_train = sm.add_constant(X_train[:,remaining_features])
        new_X_test = sm.add_constant(X_test[:,remaining_features])
        est = sm.OLS(y_train,new_X_train)
        ols_est = est.fit()
        bws_pred_y_test = ols_est.predict(new_X_test)
        bws_test_mse = mean_squared_error(y_test,bws_pred_y_test)
        bws_test_r2 = r2_score(y_test, bws_pred_y_test)
        bws_best_models[idx] = remaining_features, bws_test_mse, bws_test_r2, ols_est.params[1:], worst_feature+1
toc = time.time()
bws_time = toc - tic
best_keys = np.zeros(X.shape[1])
for i in range(0,X.shape[1]):
    best_mse = np.inf
    for key, value in bws_best_models.items():
        if (key % X.shape[1] == i):
            if value[1] < best_mse:
                best_mse = value[1]
                best_key = key
    if i == 0:
        best_keys[X.shape[1]-1] = best_key
    else:
        best_keys[i-1] = best_key
print ('\n** Backward Selection **')
print('============================================================')
print ("%-5s%-15s%-15s%-15s"%('K','MSE','R2','Subset'))
print('------------------------------------------------------------')
for i in range(X.shape[1]):
    print("%-5i%-15f%-14f"%(12-i, bws_best_models[best_keys[i]][1], bws_best_models[best_keys[i]][2]), feature_names[bws_best_models[best_keys[i]][0]].tolist())
    
print('------------------------------------------------------------')
print("Selection Algorithm: Backward Stepwise Selection")
print("Critirion: Minimizing MSE")
print("Processed in %.6f seconds"%bws_time)
print('============================================================')


#%% Lasso
tic = time.time()
lasso_mse_vec = np.zeros(N_iteration)
lasso_r2_vec = np.zeros(N_iteration)
lasso_best_mse_it = np.inf
for it in range(N_iteration):
    X_train, X_val, X_train_val, X_test, y_train, y_val, y_train_val, y_test = splitter_normalizer(X,y)
    n_alphas = 100
    alpha_vec = 10 ** np.linspace(-3,2,n_alphas)
    lasso_best_alpha = -1
    lasso_best_mse = np.inf
    for i in range(n_alphas):
        alpha = alpha_vec[i]
        lasso_regr = linear_model.Lasso(alpha=alpha)
        lasso_model = lasso_regr.fit(X_train, y_train)
        lasso_pred_y_val = lasso_model.predict(X_val)
        lasso_val_mse = mean_squared_error(y_val,lasso_pred_y_val)
        if lasso_val_mse < lasso_best_mse:
            lasso_best_mse = lasso_val_mse
            lasso_best_alpha = alpha
    lasso_regr = linear_model.Lasso(alpha=lasso_best_alpha)
    lasso_model = lasso_regr.fit(X_train_val, y_train_val)
    lasso_pred_y_test = lasso_model.predict(X_test)
    lasso_test_mse = mean_squared_error(y_test, lasso_pred_y_test)
    lasso_test_r2 = r2_score(y_test, lasso_pred_y_test)
    lasso_mse_vec[it] = lasso_test_mse
    lasso_r2_vec[it] = lasso_test_r2
    if lasso_test_mse < lasso_best_mse_it:
        lasso_best_coefs = lasso_model.coef_
        lasso_min_mse_alpha = lasso_best_alpha
toc = time.time()
lasso_time = toc - tic

print ('\n** Lasso Regularization **')
print('===================================')
print ("%-10s%-15s%-15s"%('\u03B1','MSE','R2'))
print('-----------------------------------')
print("%-10.3f%-15f%-15f"%(lasso_min_mse_alpha,np.mean(lasso_mse_vec), np.mean(lasso_r2_vec)))
print('-----------------------------------')
print("Minimum Test MSE")
print("Processed in %.6f seconds"%lasso_time)
print('===================================')


#%% Elastic Net
tic = time.time()
elnet_mse_vec = np.zeros(N_iteration)
elnet_r2_vec = np.zeros(N_iteration)
elnet_best_mse_it = np.inf
for it in range(N_iteration):
    X_train, X_val, X_train_val, X_test, y_train, y_val, y_train_val, y_test = splitter_normalizer(X,y)
    n_alphas = 100
    n_l1_rations = 10
    alpha_vec = 10 ** np.linspace(-3,-0.1,n_alphas)
    l1_ratio_vec = np.linspace(0.8,0.1,n_l1_rations,endpoint=False)[::-1][:n_l1_rations-1]
    elnet_best_mse = np.inf
    elnet_best_alpha = -1
    elnet_best_l1_ratio = -1
    for i in range(len(alpha_vec)):
        alpha = alpha_vec[i]
        for j in range(len(l1_ratio_vec)):
            l1_ratio = l1_ratio_vec[j]
            elnet_regr = linear_model.ElasticNet(alpha=alpha,l1_ratio=l1_ratio)
            elnet_model = elnet_regr.fit(X_train, y_train)
            elnet_pred_y_val = elnet_model.predict(X_val)
            elnet_val_mse = mean_squared_error(y_val,elnet_pred_y_val)
            if elnet_val_mse < elnet_best_mse:
                elnet_val_mse = elnet_val_mse
                elnet_best_l1_ratio = l1_ratio
                elnet_best_alpha = alpha
    elnet_regr = linear_model.ElasticNet(alpha=elnet_best_alpha,l1_ratio=elnet_best_l1_ratio)
    elnet_model = elnet_regr.fit(X_train_val, y_train_val)
    elnet_pred_y_test = elnet_model.predict(X_test)
    elnet_test_mse = mean_squared_error(y_test,elnet_pred_y_test)
    elnet_test_r2 = r2_score(y_test,elnet_pred_y_test)
    elnet_mse_vec[it] = elnet_test_mse
    elnet_r2_vec[it] = elnet_test_r2
    if elnet_test_mse < elnet_best_mse_it:
        elnet_best_coefs = elnet_model.coef_
        elnet_min_mse_alpha = elnet_best_alpha
        elnet_min_mse_l1_ratio = elnet_best_l1_ratio
toc = time.time()
elnet_time = toc - tic

print ('\n** Elastic Net Regularization **')
print('==========================================')
print ("%-10s%-10s%-12s%-15s"%('\u03C1','\u03B1','MSE','R2'))
print('------------------------------------------')
print("%-10.3f%-10.3f%-12f%-15f"%(elnet_min_mse_l1_ratio,elnet_min_mse_alpha,np.mean(elnet_mse_vec), np.mean(elnet_r2_vec)))
print('------------------------------------------')
print("Minimum Test MSE")
print("[\u03C1]: L1 ratio")
print("Processed in %.6f seconds"%elnet_time)
print('==========================================')

#%% Adaptive Lasso

ls_regr = linear_model.LinearRegression()
ls_model = ls_regr.fit(X_train, y_train)
ls_coefs = ls_model.coef_
warnings.filterwarnings("ignore")
tic = time.time()
alasso_mse_vec = np.zeros(N_iteration)
alasso_r2_vec = np.zeros(N_iteration)
alasso_best_mse_it = np.inf
for it in range(N_iteration):
    X_train, X_val, X_train_val, X_test, y_train, y_val, y_train_val, y_test = splitter_normalizer(X,y)
    n_lambda1 = 100
    lambda1_vec = 10 ** np.linspace(-3.5,2.5,n_lambda1)
    alasso_best_mse = np.inf
    alasso_best_lambda1 = -1
    for i in range(len(lambda1_vec)):
        lambda1 = lambda1_vec[i]
        alasso_model = asgl.ASGL(model='lm', penalization='alasso', alpha=1, lambda1=lambda1, lasso_weights=np.abs(ls_coefs))
        alasso_model.fit(x=X_train, y=y_train)
        alasso_pred_y_val = alasso_model.predict(X_val)[0]
        alasso_val_mse = mean_squared_error(y_val,alasso_pred_y_val)
        if alasso_val_mse < alasso_best_mse:
            alasso_best_mse = alasso_val_mse
            alasso_best_lambda1 = lambda1
    alasso_model = asgl.ASGL(model='lm', penalization='alasso', alpha=1, lambda1=alasso_best_lambda1, lasso_weights=np.abs(ls_coefs))
    alasso_model.fit(x=X_train_val, y=y_train_val)
    alasso_pred_y_test = alasso_model.predict(X_test)[0]
    alasso_test_mse = mean_squared_error(y_test,alasso_pred_y_test)
    alasso_test_r2 = r2_score(y_test,alasso_pred_y_test)
    alasso_mse_vec[it] = alasso_test_mse
    alasso_r2_vec[it] = alasso_test_r2
    if alasso_test_mse < alasso_best_mse_it:
        alasso_best_coefs = alasso_model.coef_[0][1:]
        alasso_min_mse_lambda1 = alasso_best_lambda1
toc = time.time()
alasso_time = toc - tic

print ('\n** Adaptive Lasso Regularization **')
print('===================================')
print ("%-10s%-15s%-15s"%('\u03B1','MSE','R2'))
print('-----------------------------------')
print("%-10.3f%-15f%-15f"%(alasso_min_mse_lambda1,np.mean(alasso_mse_vec), np.mean(alasso_r2_vec)))
print('-----------------------------------')
print("Minimum Test MSE")
print("Processed in %.6f seconds"%alasso_time)
print('===================================')


#%% Final Evaluation

print ('\n%-15s%-10s'%(' ','** Coefficients in Different Linear Models **'))
print('=================================================================================')
print ("%-10s%-15s%-15s%-15s%-15s%-15s"%('Feature','OLS','Ridge','Lasso','ElNet','ALasso'))
print('---------------------------------------------------------------------------------')
for i in range(X.shape[1]):
    print("%-10s%-15f%-15f%-15f%-15f%-15f"%(feature_names[i],ls_best_coefs[i],ridge_best_coefs[i],lasso_best_coefs[i],elnet_best_coefs[i],alasso_best_coefs[i]))
print('---------------------------------------------------------------------------------')
print("Criterion: Minimum Test MSE")
print('=================================================================================')












