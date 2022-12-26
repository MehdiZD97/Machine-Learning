# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 00:25:34 2022

@author: mz52
"""

from tree_methods_lib import *
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
import time
import warnings
warnings.filterwarnings('ignore')

# %% Preprocessing

# load train and test datasets
data_train, data_test, features = load_dataset(print_info=False)
# prepare datasets for classification
X_train, X_test, Y_train, Y_test = preparing_dataset(data_train, data_test)



#%% AdaBoost - max tree depth

print('\n> Changing Hyperparameters with AdaBoost Classifier')
learning_rate = 0.5
num_trees = 100
max_depth_vec = [1, 5, 10, 15, 20, 25]
print('... Learning rate: %.2f'%learning_rate)
print('... Num trees: %i'%num_trees)
print('... Max depth range: {} to {}'.format(np.min(max_depth_vec),np.max(max_depth_vec)))
loss_funcs = ['gini','entropy']
print('... Criterions: ',loss_funcs)
acc_vec = np.zeros((len(max_depth_vec),2,2),dtype=float)
mis_clf_vec = np.zeros((len(max_depth_vec),2,2),dtype=float)
for crt in range(len(loss_funcs)):
    for i in range(len(max_depth_vec)):
        print('... %s with max depth = %i'%(loss_funcs[crt],max_depth_vec[i]))
        max_depth = max_depth_vec[i]
        base_clf = DecisionTreeClassifier(criterion=loss_funcs[crt], splitter='best', max_depth=max_depth)
        ab_clf = AdaBoostClassifier(base_estimator=base_clf, n_estimators=num_trees, learning_rate=learning_rate, random_state=5)
        ab_clf.fit(X_train, Y_train)
        Y_pred_train = ab_clf.predict(X_train)
        Y_pred_test = ab_clf.predict(X_test)
        acc_vec[i,0,crt] = metrics.accuracy_score(Y_train, Y_pred_train)
        acc_vec[i,1,crt] = metrics.accuracy_score(Y_test, Y_pred_test)
        mis_clf_vec[i,0,crt] = np.sum(np.abs(Y_pred_train-Y_train.to_numpy()))/len(Y_train)
        mis_clf_vec[i,1,crt] = np.sum(np.abs(Y_pred_test-Y_test.to_numpy()))/len(Y_test)

#%% Result - max tree depth

print('... Results')
plt.rcdefaults()
plt.figure(figsize=(20,8))
plt.rc('font',size=18)
plt.subplot(1,2,1)
plt.plot(max_depth_vec, acc_vec[:,0,0], '-o', lw=2, color='b', label='Gini/Train')
plt.plot(max_depth_vec, acc_vec[:,1,0], '-o', lw=2, color='r', label='Gini/Test')
plt.plot(max_depth_vec, acc_vec[:,0,1], '--s', lw=2, color='b', label='CE/Train')
plt.plot(max_depth_vec, acc_vec[:,1,1], '--s', lw=2, color='r', label='CE/Test')
plt.legend(loc='best',fontsize=18)
plt.grid(axis='y',alpha=0.5)
plt.xticks(max_depth_vec)
plt.xlabel('Max Tree Depth')
plt.ylabel('Accuracy')
plt.subplot(1,2,2)
plt.plot(max_depth_vec, mis_clf_vec[:,0,0], '-o', lw=2, color='b', label='Gini/Train')
plt.plot(max_depth_vec, mis_clf_vec[:,1,0], '-o', lw=2, color='r', label='Gini/Test')
plt.plot(max_depth_vec, mis_clf_vec[:,0,1], '--s', lw=2, color='b', label='CE/Train')
plt.plot(max_depth_vec, mis_clf_vec[:,1,1], '--s', lw=2, color='r', label='CE/Test')
plt.legend(loc='best',fontsize=18)
plt.grid(axis='y',alpha=0.5)
plt.xticks(max_depth_vec)
plt.xlabel('Max Tree Depth')
plt.ylabel('Misclassification Error')
plt.suptitle('AdaBoost Train-Test Accuracy / Error')
plt.show()



#%% AdaBoost - learning rate

print('\n> Changing Hyperparameters with AdaBoost Classifier')
learning_rate_vec = [0.01, 0.05, 0.1, 0.25, 0.5, 0.8, 1]
num_trees = 100
max_depth = 5
print('... Learning rate range: {} to {}'.format(np.min(learning_rate_vec),np.max(learning_rate_vec)))
print('... Num trees: %i'%num_trees)
print('... Max depth: {}'.format(max_depth))
loss_funcs = ['gini','entropy']
print('... Criterions: ',loss_funcs)
acc_vec = np.zeros((len(learning_rate_vec),2,2),dtype=float)
mis_clf_vec = np.zeros((len(learning_rate_vec),2,2),dtype=float)
for crt in range(len(loss_funcs)):
    for i in range(len(learning_rate_vec)):
        print('... %s with learning rate = %.2f'%(loss_funcs[crt],learning_rate_vec[i]))
        learning_rate = learning_rate_vec[i]
        base_clf = DecisionTreeClassifier(criterion=loss_funcs[crt], splitter='best', max_depth=max_depth)
        ab_clf = AdaBoostClassifier(base_estimator=base_clf, n_estimators=num_trees, learning_rate=learning_rate, random_state=5)
        ab_clf.fit(X_train, Y_train)
        Y_pred_train = ab_clf.predict(X_train)
        Y_pred_test = ab_clf.predict(X_test)
        acc_vec[i,0,crt] = metrics.accuracy_score(Y_train, Y_pred_train)
        acc_vec[i,1,crt] = metrics.accuracy_score(Y_test, Y_pred_test)
        mis_clf_vec[i,0,crt] = np.sum(np.abs(Y_pred_train-Y_train.to_numpy()))/len(Y_train)
        mis_clf_vec[i,1,crt] = np.sum(np.abs(Y_pred_test-Y_test.to_numpy()))/len(Y_test)

#%% Result - learning rate

print('... Results')
plt.rcdefaults()
plt.figure(figsize=(20,8))
plt.rc('font',size=18)
plt.subplot(1,2,1)
plt.plot(learning_rate_vec, acc_vec[:,0,0], '-o', lw=2, color='b', label='Gini/Train')
plt.plot(learning_rate_vec, acc_vec[:,1,0], '-o', lw=2, color='r', label='Gini/Test')
plt.plot(learning_rate_vec, acc_vec[:,0,1], '--s', lw=2, color='b', label='CE/Train')
plt.plot(learning_rate_vec, acc_vec[:,1,1], '--s', lw=2, color='r', label='CE/Test')
plt.legend(loc='best',fontsize=18)
plt.grid(axis='y',alpha=0.5)
plt.xticks([0.01, 0.1, 0.25, 0.5, 0.8, 1])
plt.xlabel('Learning rate')
plt.ylabel('Accuracy')
plt.subplot(1,2,2)
plt.plot(learning_rate_vec, mis_clf_vec[:,0,0], '-o', lw=2, color='b', label='Gini/Train')
plt.plot(learning_rate_vec, mis_clf_vec[:,1,0], '-o', lw=2, color='r', label='Gini/Test')
plt.plot(learning_rate_vec, mis_clf_vec[:,0,1], '--s', lw=2, color='b', label='CE/Train')
plt.plot(learning_rate_vec, mis_clf_vec[:,1,1], '--s', lw=2, color='r', label='CE/Test')
plt.legend(loc='best',fontsize=18)
plt.grid(axis='y',alpha=0.5)
plt.xticks([0.01, 0.1, 0.25, 0.5, 0.8, 1])
plt.xlabel('Learning rate')
plt.ylabel('Misclassification Error')
plt.suptitle('AdaBoost Train-Test Accuracy / Error')
plt.show()


#%% Grid Search CV

print('\n> Grid Search Cross-Validation for Bagging')
tic = time.time()
model_cv = AdaBoostClassifier(n_estimators=100, learning_rate=0.01)
n_trees = [100, 500, 1000, 5000]
learning_rate = [0.01, 0.05, 0.1, 0.25, 0.5]
param_grid = dict(n_estimators=n_trees, learning_rate=learning_rate)
grid = GridSearchCV(estimator=model_cv, param_grid=param_grid, cv=3, verbose=3)
grid_result = grid.fit(X_train, Y_train)
grid_best_params = grid.best_params_
toc = time.time()
bg_grid_time = toc - tic

print("\nprocessing time: %.6f sec\n"%bg_grid_time)
print(f'Best Accuracy for {grid_result.best_score_:.4} using {grid_result.best_params_}')
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(f'mean={mean:.4}, std={stdev:.4} using {param}')

