# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 17:27:25 2022

@author: mz52
"""

from sklearn.datasets import load_boston
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from yellowbrick.regressor import PredictionError, ResidualsPlot
import warnings
from itertools import combinations
import time
import asgl


#%% Preprocessing

#Loading datasets
"""
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
X = data
target = raw_df.values[1::2, 2]
y = target
"""
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

# Splitting
test_size = 0.25   # Test portion of the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
print("%-24s %i%%"%('Training proportion:',(1-test_size)*100))
print("%-24s %i%%"%('Test proportion:',test_size*100))

# Standardization
std_scaler = StandardScaler()
X_train = std_scaler.fit_transform(X_train)
X_test = std_scaler.transform(X_test)

#%% Linear Regression
print("\n** Linear Regression **")
ls_visualization = True
tic = time.time()
ls_regr = linear_model.LinearRegression()
ls_model = ls_regr.fit(X_train, y_train)
ls_intercept = ls_model.intercept_
ls_coef = ls_model.coef_

ls_pred_y_train = ls_model.predict(X_train)
ls_pred_y_test = ls_model.predict(X_test)

ls_train_mse = mean_squared_error(y_train, ls_pred_y_train)
ls_train_r2 = r2_score(y_train, ls_pred_y_train)
ls_test_mse = mean_squared_error(y_test, ls_pred_y_test)
ls_test_r2 = r2_score(y_test, ls_pred_y_test)
toc = time.time()

# Model Evaluation
print('\n===================================')
print ("%-10s%-15s%-15s"%(' ','MSE','R2'))
print('-----------------------------------')
print("%-10s%-15f%-15f"%('Training', ls_train_mse, ls_train_r2))
print("%-10s%-15f%-15f"%('Test', ls_test_mse, ls_test_r2))
print('-----------------------------------')
print("Linear Model: Least Square")
print("Processed in %.6f seconds"%(toc-tic))
print('===================================')

const_added_X_train = sm.add_constant(X_train)
est = sm.OLS(y_train, const_added_X_train)
ols_est = est.fit()
print("\n",ols_est.summary())

# Visualization
if (ls_visualization):
    
    plt.figure(figsize=(12,8))
    ax = sns.scatterplot(x=y_train, y=ls_pred_y_train)
    ax = sns.scatterplot(x=y_test, y=ls_pred_y_test)
    ax.legend(["Training", "Test"], fontsize=18, loc='upper left')
    ax.set_xlabel('Actual Response', fontsize=14)
    ax.set_ylabel('Predicted Response', fontsize=14)
    perfect_pred = np.arange(0,51)
    ax = plt.plot(perfect_pred, perfect_pred, '--', c='black', alpha=0.65)
    
    plt.figure(figsize=(10,8))
    ls_visualizer = PredictionError(ls_model)
    ls_visualizer.score(X_train, y_train)
    ls_visualizer.score(X_test, y_test)
    ls_visualizer.show()
    
    plt.figure(figsize=(10,6))
    ls_visualizer = ResidualsPlot(ls_model)
    ls_visualizer.fit(X_train, y_train)
    ls_visualizer.score(X_test, y_test)
    ls_visualizer.show()

#%% Best Subsets RSS
bss_min_mse = True
feature_idxs = np.arange(X.shape[1])
def processSubset(selected_features):
    # Fit model on selected_features and calculate RSS   
    new_X_train = X_train[:, selected_features]
    const_added_X_train = sm.add_constant(new_X_train)
    est = sm.OLS(y_train, const_added_X_train)
    ols_est = est.fit()
    return ols_est.ssr, ols_est.rsquared, ols_est.aic

if(bss_min_mse):
    print("\n** Best Subset Selection **")
    tic = time.time()    
    best_models = {}
    for i in range(1, X.shape[1]+1):
        best_rss = np.inf
        for comb in combinations(feature_idxs, i):
            selected_features = np.asarray(comb)
            rss, r2, aic = processSubset(selected_features)
            if rss < best_rss:
                best_rss = rss
                best_models[i] = selected_features, best_rss, r2, aic
    
    toc = time.time()
    print('\n============================================================')
    print ("%-5s%-15s%-15s%-15s%-15s"%('K','RSS','R2','AIC','Subset'))
    print('------------------------------------------------------------')
    for key, value in best_models.items():
        print("%-5i%-15f%-15f%-14f"%(key, value[1], value[2], value[3]), feature_names[value[0]].tolist())
        
    print('------------------------------------------------------------')
    print("Selection Algorithm: Exhaustive")
    print("Critirion: Minimizing RSS")
    print("Processed in %.6f seconds"%(toc-tic))
    print('============================================================')

#%% Best Subsets AIC
bss_min_aic = True
if(bss_min_aic):
    print("\n** Best Subset Selection **")
    tic = time.time()    
    best_models = {}
    for i in range(1, X.shape[1]+1):
        best_aic = np.inf
        for comb in combinations(feature_idxs, i):
            selected_features = np.asarray(comb)
            rss, r2, aic = processSubset(selected_features)
            if aic < best_aic:
                best_aic = aic
                best_models[i] = selected_features, rss, r2, best_aic
    
    toc = time.time()
    print('\n============================================================')
    print ("%-5s%-15s%-15s%-15s%-15s"%('K','RSS','R2','AIC','Subset'))
    print('------------------------------------------------------------')
    for key, value in best_models.items():
        print("%-5i%-15f%-15f%-14f"%(key, value[1], value[2], value[3]), feature_names[value[0]].tolist())
        
    print('------------------------------------------------------------')
    print("Selection Algorithm: Exhaustive")
    print("Critirion: Minimizing AIC")
    print("Processed in %.6f seconds"%(toc-tic))
    print('============================================================')

    
#%% Forward Stepwise AIC
fss_activate = True
if(fss_activate):
    print("\n** Forward Stepwise Selection **")
    tic = time.time()
    remaining_features = feature_idxs
    selected_features = np.array([],dtype=int)
    best_models = {}
    for i in range(1, X.shape[1]+1):
        best_rss = np.inf
        best_r2 = 0
        best_aic = np.inf
        for comb in combinations(remaining_features, 1):
            temp_feature = np.asarray(comb)
            rss, r2, aic = processSubset(np.append(selected_features,temp_feature))
            if aic < best_aic:
                best_aic = aic
                best_rss = rss
                best_r2 = r2
                best_feature = temp_feature
        selected_features = np.append(selected_features,best_feature)
        best_models[i] = selected_features, best_rss, best_r2, best_aic, best_feature+1
        remaining_features = np.delete(remaining_features, np.where(remaining_features==best_feature)[0][0])
    
    toc = time.time()
    print('\n============================================================')
    print ("%-5s%-15s%-15s%-15s%-15s"%('(+)','RSS','R2','AIC','Subset'))
    print('------------------------------------------------------------')
    for key, value in best_models.items():
        print("%-5i%-15f%-15f%-14f"%(value[4], value[1], value[2], value[3]), feature_names[value[0]].tolist())
        
    print('------------------------------------------------------------')
    print("Selection Algorithm: Forward Stepwise Selection")
    print("Critirion: Minimizing AIC")
    print("Processed in %.6f seconds"%(toc-tic))
    print('============================================================')

#%% Backward Stepwise AIC
bss_activate = True
if(bss_activate):
    print("\n** Backward Stepwise Selection **")
    tic = time.time()
    remaining_features = feature_idxs
    best_models = {}
    for i in range(1, X.shape[1]+1):
        best_rss = np.inf
        best_r2 = 0
        best_aic = np.inf
        for comb in combinations(remaining_features, 1):
            temp_feature = np.asarray(comb)
            rss, r2, aic = processSubset(np.delete(remaining_features,np.where(remaining_features==temp_feature)[0][0]))
            if aic < best_aic:
                best_aic = aic
                best_rss = rss
                best_r2 = r2
                worst_feature = temp_feature
        remaining_features = np.delete(remaining_features, np.where(remaining_features==worst_feature)[0][0])
        best_models[i] = remaining_features, best_rss, best_r2, best_aic, worst_feature+1
    
    toc = time.time()
    print('\n============================================================')
    print ("%-5s%-15s%-15s%-15s%-15s"%('(-)','RSS','R2','AIC','Subset'))
    print('------------------------------------------------------------')
    for key, value in best_models.items():
        print("%-5i%-15f%-15f%-14f"%(value[4], value[1], value[2], value[3]), feature_names[value[0]].tolist())
        
    print('------------------------------------------------------------')
    print("Selection Algorithm: Backward Stepwise Selection")
    print("Critirion: Minimizing AIC")
    print("Processed in %.6f seconds"%(toc-tic))
    print('============================================================')

#%% Lasso with sklearn
print("\n** Lasso Regression **")
tic = time.time()
alpha_vec = np.array([0.1, 0.5, 1, 3, 5])
lasso_rss_vec = np.zeros(len(alpha_vec)) - 1
lasso_r2_vec = np.zeros(len(alpha_vec)) - 1
all_lasso_coefs = {}
for i in range(len(alpha_vec)):
    alpha = alpha_vec[i]
    lasso_regr = linear_model.Lasso(alpha=alpha)
    lasso_model = lasso_regr.fit(X_train, y_train)
    all_lasso_coefs[i] = lasso_model.coef_
    lasso_pred_y_train = lasso_model.predict(X_train)
    lasso_rss = np.sum((y_train - lasso_pred_y_train)**2)
    lasso_rss_vec[i] = lasso_rss
    lasso_tss = np.sum((y_train - np.mean(y_train))**2)
    lasso_r2 = 1 - lasso_rss/lasso_tss
    lasso_r2_vec[i] = lasso_r2
toc = time.time()
# Model Evaluation
print('\n====================================')
print ("%-10s%-15s%-15s"%('Alpha','RSS','R2'))
print('------------------------------------')
for i in range(len(alpha_vec)):
    print("%-10.2f%-15f%-15f"%(alpha_vec[i], lasso_rss_vec[i], lasso_r2_vec[i]))
print('------------------------------------')
print("Linear Model: Lasso Regularization")
print("Processed in %.6f seconds"%(toc-tic))
print('====================================')

lasso_coef_mat = np.zeros((X.shape[1],len(alpha_vec)))
for key, value in all_lasso_coefs.items():
    lasso_coef_mat[:,key] = value
print('\n============================================================')
print ("%-9s \u03B1=%-7.2f \u03B1=%-7.2f \u03B1=%-7.2f \u03B1=%-7.2f \u03B1=%-7.2f"%('Feature',alpha_vec[0],alpha_vec[1],alpha_vec[2],alpha_vec[3],alpha_vec[4]))
print('------------------------------------------------------------')
for i in range(X.shape[1]):
    print("%-10s%-10.4f%-10.4f%-10.4f%-10.4f%-10.4f"%(feature_names[i], lasso_coef_mat[i,0], lasso_coef_mat[i,1], lasso_coef_mat[i,2], lasso_coef_mat[i,3], lasso_coef_mat[i,4]))
    
print('------------------------------------------------------------')
print("Linear Model: Lasso Regularization")
print("Coefficients of the linear model")
print('============================================================')


# print np.abs of lasso_model.coef_ to show the importance of each feature

#%% Lasso statsmodels
lasso_sm_activate = False
if (lasso_sm_activate):
    print("\n** Lasso Regression **")
    tic = time.time()
    alpha_vec = np.array([0.1, 0.5, 1, 5, 10])
    lasso_rss_vec = np.zeros(len(alpha_vec)) - 1
    lasso_r2_vec = np.zeros(len(alpha_vec)) - 1
    L1_wt = 1   # If 0, fit is Ridge & if 1, fit is Lasso & if (0,1) the fit is ElasticNet
    const_added_X_train = sm.add_constant(X_train)
    for i in range(len(alpha_vec)):
        alpha = alpha_vec[i]
        est = sm.OLS(y_train, const_added_X_train)
        lasso_est = est.fit_regularized(method='elastic_net', alpha=alpha, L1_wt=L1_wt)
        lasso_pred_y_train = lasso_est.fittedvalues
        lasso_rss = np.sum((y_train - lasso_pred_y_train)**2)
        lasso_rss_vec[i] = lasso_rss
        lasso_tss = np.sum((y_train - np.mean(y_train))**2)
        lasso_r2 = 1 - lasso_rss/lasso_tss
        lasso_r2_vec[i] = lasso_r2
    toc = time.time()
    # Model Evaluation
    print('\n====================================')
    print ("%-10s%-15s%-15s"%('Alpha','RSS','R2'))
    print('------------------------------------')
    for i in range(len(alpha_vec)):
        print("%-10.2f%-15f%-15f"%(alpha_vec[i], lasso_rss_vec[i], lasso_r2_vec[i]))
    print('------------------------------------')
    print("Linear Model: Lasso Regularization")
    print("Processed in %.6f seconds"%(toc-tic))
    print('====================================')


#%% ElasticNet with sklearn
print("\n** Elastic Net Regression **")
tic = time.time()
alpha_vec = np.array([0.1, 0.5, 1, 5])
l1_ratio_vec = np.array([0.1, 0.5, 0.8])
en_rss_vec = np.zeros(len(alpha_vec)*len(l1_ratio_vec)) - 1
en_r2_vec = np.zeros(len(alpha_vec)*len(l1_ratio_vec)) - 1
all_en_coefs = {}
for i in range(len(alpha_vec)):
    alpha = alpha_vec[i]
    for j in range(len(l1_ratio_vec)):
        l1_ratio = l1_ratio_vec[j]
        idx = i*len(l1_ratio_vec) + j
        en_regr = linear_model.ElasticNet(alpha=alpha,l1_ratio=l1_ratio)
        en_model = en_regr.fit(X_train, y_train)
        all_en_coefs[idx] = en_model.coef_
        en_pred_y_train = en_model.predict(X_train)
        en_rss = np.sum((y_train - en_pred_y_train)**2)
        en_rss_vec[idx] = en_rss
        en_tss = np.sum((y_train - np.mean(y_train))**2)
        en_r2 = 1 - en_rss/en_tss
        en_r2_vec[idx] = en_r2
toc = time.time()
# Model Evaluation
print('\n=============================================')
print ("%-8s%-12s%-15s%-15s"%('Alpha','L1_ratio','RSS','R2'))
print('---------------------------------------------')
for i in range(len(alpha_vec)):
    for j in range(len(l1_ratio_vec)):
        idx = i*len(l1_ratio_vec) + j
        print("%-8.2f%-12.2f%-15f%-15f"%(alpha_vec[i], l1_ratio_vec[j], en_rss_vec[idx], en_r2_vec[idx]))
    print('---------------------------------------------')
print("Linear Model: Elastic Net Regularization")
print("Processed in %.6f seconds"%(toc-tic))
print('=============================================')

en_coef_mat = np.zeros((X.shape[1],len(alpha_vec)*len(l1_ratio_vec)))
for key, value in all_en_coefs.items():
    en_coef_mat[:,key] = value
print('\n================================================================================================================================')
print ("%-9s \u03B1=%-7.2f \u03B1=%-7.2f \u03B1=%-7.2f \u03B1=%-7.2f \u03B1=%-7.2f \u03B1=%-7.2f \u03B1=%-7.2f \u03B1=%-7.2f \u03B1=%-7.2f \u03B1=%-7.2f \u03B1=%-7.2f \u03B1=%-7.2f"%('Feature',alpha_vec[0],alpha_vec[0],alpha_vec[0],alpha_vec[1],alpha_vec[1],alpha_vec[1],alpha_vec[2],alpha_vec[2],alpha_vec[2],alpha_vec[3],alpha_vec[3],alpha_vec[3]))
print ("%-9s \u03C1=%-7.2f \u03C1=%-7.2f \u03C1=%-7.2f \u03C1=%-7.2f \u03C1=%-7.2f \u03C1=%-7.2f \u03C1=%-7.2f \u03C1=%-7.2f \u03C1=%-7.2f \u03C1=%-7.2f \u03C1=%-7.2f \u03C1=%-7.2f"%('Names',l1_ratio_vec[0],l1_ratio_vec[1],l1_ratio_vec[2],l1_ratio_vec[0],l1_ratio_vec[1],l1_ratio_vec[2],l1_ratio_vec[0],l1_ratio_vec[1],l1_ratio_vec[2],l1_ratio_vec[0],l1_ratio_vec[1],l1_ratio_vec[2]))
print('--------------------------------------------------------------------------------------------------------------------------------')
for i in range(X.shape[1]):
    print("%-10s%-10.4f%-10.4f%-10.4f%-10.4f%-10.4f%-10.4f%-10.4f%-10.4f%-10.4f%-10.4f%-10.4f%-10.4f"%(feature_names[i],en_coef_mat[i,0],en_coef_mat[i,1],en_coef_mat[i,2],en_coef_mat[i,3],en_coef_mat[i,4],en_coef_mat[i,5],en_coef_mat[i,6],en_coef_mat[i,7],en_coef_mat[i,8],en_coef_mat[i,9],en_coef_mat[i,10],en_coef_mat[i,11]))
    
print('--------------------------------------------------------------------------------------------------------------------------------')
print("Linear Model: Elastic Net Regularization")
print("Coefficients of the linear model")
print("[\u03C1]: L1 ratio")
print('================================================================================================================================')


#%% Adaptive Lasso asgl
print("\n** Adaptive Lasso Regression **")
tic = time.time()
lambda1 = np.array([0.01, 0.1, 0.5, 1, 5])
alasso_model = asgl.ASGL(model='lm', penalization='alasso', alpha=1, lambda1=lambda1, lasso_weights=np.abs(ls_coef))
alasso_model.fit(x=X_train, y=y_train)
alasso_coefs = alasso_model.coef_
alasso_pred_y_train = alasso_model.predict(X_train)
alasso_rss_vec = np.zeros(len(lambda1))
alasso_r2_vec = np.zeros(len(lambda1))
for i in range(len(lambda1)):
    alasso_rss = np.sum((y_train - alasso_pred_y_train[i])**2)
    alasso_rss_vec[i] = alasso_rss
    alasso_tss = np.sum((y_train - np.mean(y_train))**2)
    alasso_r2 = 1 - alasso_rss/alasso_tss
    alasso_r2_vec[i] = alasso_r2
toc = time.time()
# Model Evaluation
print('\n============================================')
print ("%-12s%-17s%-15s"%('\u03BB','RSS','R2'))
print('--------------------------------------------')
for i in range(len(lambda1)):
    print("%-12.2f%-17f%-15f"%(lambda1[i], alasso_rss_vec[i], alasso_r2_vec[i]))
print('--------------------------------------------')
print("Linear Model: Adaptive Lasso Regularization")
print("Weights: OLS Unbiased Estimates")
print("Processed in %.6f seconds"%(toc-tic))
print('============================================')

alasso_coef_mat = np.zeros((X.shape[1],len(lambda1)))
for i in range(len(lambda1)):
    alasso_coef_mat[:,i] = alasso_coefs[i][1:]
print('\n============================================================')
print ("%-10s \u03BB=%-7.2f \u03BB=%-7.2f \u03BB=%-7.2f \u03BB=%-7.2f \u03BB=%-7.2f"%('Feature',lambda1[0],lambda1[1],lambda1[2],lambda1[3],lambda1[4]))
print('------------------------------------------------------------')
for i in range(X.shape[1]):
    print("%-11s%-10.4f%-10.4f%-10.4f%-10.4f%-10.4f"%(feature_names[i],alasso_coef_mat[i,0],alasso_coef_mat[i,1],alasso_coef_mat[i,2],alasso_coef_mat[i,3],alasso_coef_mat[i,4]))
    
print('------------------------------------------------------------')
print("Linear Model: Adaptive Lasso Regularization")
print("Coefficients of the linear model")
print('============================================================')


#%% Path Visualization
path_visualization = True
if (path_visualization):
    # Lasso
    alpha = 0.1
    lasso_regr = linear_model.Lasso(alpha=alpha)
    lasso_model = lasso_regr.fit(X_train, y_train)
    lasso_path = lasso_model.path(X_train,y_train,l1_ratio=1)
    plt.figure(figsize=(12,8))
    for i in range(X.shape[1]):
        plt.plot(-np.log(lasso_path[0]),lasso_path[1][i,:],label=feature_names[i],lw=2)
    plt.grid(False)
    plt.ylim([-5,5])
    plt.legend(fontsize=10,loc='upper left')
    plt.xlabel('-Log(\u03B1)',fontsize=14)
    plt.ylabel('Coefficients',fontsize=14)
    plt.title('Lasso Regularization Paths',fontsize=14)
    
    # Elastic Net
    l1_ratio_vec = np.array([0.75,0.5,0.25])
    for l1_r in range(len(l1_ratio_vec)):
        l1_ratio = l1_ratio_vec[l1_r]
        en_regr = linear_model.ElasticNet(alpha=alpha,l1_ratio=l1_ratio)
        en_model = en_regr.fit(X_train, y_train)
        en_path = en_model.path(X_train,y_train,l1_ratio=l1_ratio)
        plt.figure(figsize=(12,8))
        for i in range(X.shape[1]):
            plt.plot(-np.log(en_path[0]),en_path[1][i,:],label=feature_names[i],lw=2)
        plt.grid(False)
        plt.ylim([-5,5])
        plt.legend(fontsize=10,loc='upper left')
        plt.xlabel('-Log(\u03B1)',fontsize=14)
        plt.ylabel('Coefficients',fontsize=14)
        plt.title('Elastic Net Regularization Paths with L1-Ratio = %.2f'%l1_ratio,fontsize=14)
    
    # Ridge
    n_alphas = 100
    alpha_vec = 10 ** np.linspace(-0.5,4.5,n_alphas)
    ridge_coefs = np.zeros((X.shape[1],n_alphas)) - np.inf
    for i in range(n_alphas):
        alpha = alpha_vec[i]
        ridge_regr = linear_model.Ridge(alpha=alpha)
        ridge_model = ridge_regr.fit(X_train, y_train)
        ridge_coefs[:,i] = ridge_model.coef_
    plt.figure(figsize=(12,8))
    for i in range(X.shape[1]):
        plt.plot(-np.log(alpha_vec),ridge_coefs[i,:],label=feature_names[i],lw=2)
    plt.grid(False)
    plt.ylim([-5,5])
    plt.legend(fontsize=10,loc='upper left')
    plt.xlabel('-Log(\u03B1)',fontsize=14)
    plt.ylabel('Coefficients',fontsize=14)
    plt.title('Ridge Regularization Paths',fontsize=14)
    plt.show()



