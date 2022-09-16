# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 23:15:12 2022

@author: mz52
"""

from sklearn.datasets import load_boston
import numpy as np
import statsmodels.api as sm
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
from itertools import combinations
import time
import asgl


#%% Preprocessing

#Loading datasets
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    boston_dataset = load_boston()
X = boston_dataset.data
y = boston_dataset.target
feature_names = boston_dataset.feature_names
print("** Empirical Dataset **\n")
print("%-25s%-25s"%('Dataset:','Boston House Prices'))
print("%-24s"%('Dataset dimensions:'), X.shape)
print("%-24s"%('Feature names:'), feature_names.tolist())

# Splitting
test_size = 0.25   # Test portion of the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
print("%-24s %i%%"%('Training proportion:',(1-test_size)*100))
print("%-24s %i%%"%('Test proportion:',test_size*100))

#%% Centring vs 1-Column

X_train_mean = np.tile(np.reshape(np.mean(X_train,axis=0),(1,X_train.shape[1])),(X_train.shape[0],1))
X_train_centered = X_train - X_train_mean
y_train_centered = y_train - np.mean(y_train)
ls_regr = linear_model.LinearRegression()
ls_model = ls_regr.fit(X_train_centered, y_train_centered)
model_centered_coefs = ls_model.coef_

X_train_1col = np.ones((X_train.shape[0],X_train.shape[1]+1))
X_train_1col[:,1:] = X_train
ls_regr = linear_model.LinearRegression()
ls_model = ls_regr.fit(X_train_1col, y_train)
model_1col_coefs = ls_model.coef_

print ('\n%-5s%-10s'%(' ','** Coefficients of Linear Model **'))
print('==============================================')
print ("%-14s%-17s%-17s"%('Feature','Centered','1-Column'))
print('----------------------------------------------')
for i in range(X.shape[1]):
    print("%-14s%-17f%-17f"%(feature_names[i], model_centered_coefs[i], model_1col_coefs[i+1]))
print('----------------------------------------------')
print("Take Away: Same Result (equivalent methods)")
print('==============================================')


#%% High-dimensional Data

hd_X, hd_y = datasets.make_regression(n_samples=100, n_features=500, noise=0.01, random_state=1)

# Splitting
hd_test_size = 0.2   # Test portion of the dataset
hd_X_train, hd_X_test, hd_y_train, hd_y_test = train_test_split(hd_X, hd_y, test_size=hd_test_size, random_state=1)
print('\n=========================================================')
print("%-25s%-25s"%('Dataset:','Random High-Dimensional Dataset'))
print("%-24s"%('Dataset dimensions:'), hd_X.shape)
print("%-24s %i%%"%('Training proportion:',(1-hd_test_size)*100))
print("%-24s %i%%"%('Test proportion:',hd_test_size*100))

# Adding 1-Column
hd_X_train_1col = np.ones((hd_X_train.shape[0],hd_X_train.shape[1]+1))
hd_X_train_1col[:,1:] = hd_X_train

# Least Square
ls_regr = linear_model.LinearRegression()
ls_model = ls_regr.fit(hd_X_train_1col, hd_y_train)
pred_hd_y_train = ls_model.predict(hd_X_train_1col)
training_error = np.sum((pred_hd_y_train - hd_y_train) ** 2)
print('---------------------------------------------------------')
print("Least Square Training Error:")
print(training_error)
print('---------------------------------------------------------')
print("Take Away: OLS has zero training error when p > n")
print('=========================================================')


#%% MSE Existence Theorem

# Least Square
X_train_1col = np.ones((X_train.shape[0],X_train.shape[1]+1))
X_train_1col[:,1:] = X_train
X_test_1col = np.ones((X_test.shape[0],X_test.shape[1]+1))
X_test_1col[:,1:] = X_test
ls_regr = linear_model.LinearRegression()
ls_model = ls_regr.fit(X_train_1col, y_train)
ls_pred_y_test = ls_model.predict(X_test_1col)
ls_test_mse = mean_squared_error(y_test,ls_pred_y_test)


# Ridge
n_alphas = 100
alpha_vec = 10 ** np.linspace(-3,1,n_alphas)
ridge_best_alpha = -1
ridge_best_mse = ls_test_mse
for i in range(n_alphas):
    alpha = alpha_vec[i]
    ridge_regr = linear_model.Ridge(alpha=alpha)
    ridge_model = ridge_regr.fit(X_train_1col, y_train)
    ridge_pred_y_test = ridge_model.predict(X_test_1col)
    ridge_test_mse = mean_squared_error(y_test, ridge_pred_y_test)
    if ridge_test_mse < ridge_best_mse:
        ridge_best_mse = ridge_test_mse
        ridge_best_alpha = alpha

print ('\n** MSE Existence Theorem **')
print('===================================')
print ("%-10s%-15s%-15s"%('\u03BB','Ridge','OLS'))
print('-----------------------------------')
print("%-10.4f%-15f%-15f"%(ridge_best_alpha,ridge_best_mse,ls_test_mse))
print('-----------------------------------')
print("Minimizing Test MSE")
print('===================================')


