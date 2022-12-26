# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 18:46:31 2022

@author: mz52
"""

from tree_methods_lib import *
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
import time
import warnings
warnings.filterwarnings('ignore')

# %% Preprocessing

# load train and test datasets
data_train, data_test, features = load_dataset(print_info=False)
# prepare datasets for classification
X_train, X_test, Y_train, Y_test = preparing_dataset(data_train, data_test)


#%% Random Forest

print('\n> Finding best num trees for Random Forest Classifier')
num_trees = 10**np.arange(1,6)
#num_trees = [10, 20, 30, 40, 50]
max_depth = 10
print('... Num trees range: {} to {}'.format(np.min(num_trees),np.max(num_trees)))
print('... Max depth: %i'%max_depth)
loss_funcs = ['gini','entropy']
print('... Criterions: ',loss_funcs)
acc_vec = np.zeros((len(num_trees),2,2),dtype=float)
log_loss_vec = np.zeros((len(num_trees),2,2),dtype=float)
mis_clf_vec = np.zeros((len(num_trees),2,2),dtype=float)
for crt in range(len(loss_funcs)):
    for i in range(len(num_trees)):
        print('... %s with num trees = %i'%(loss_funcs[crt],num_trees[i]))
        num_est = num_trees[i]
        rf_clf = RandomForestClassifier(n_estimators=num_est, criterion=loss_funcs[crt], max_depth=max_depth, bootstrap=True, oob_score=True, random_state=5)
        rf_clf.fit(X_train, Y_train)
        Y_pred_train = rf_clf.predict(X_train)
        Y_pred_test = rf_clf.predict(X_test)
        Y_pred_prob_train = rf_clf.predict_proba(X_train)
        Y_pred_prob_test = rf_clf.predict_proba(X_test)
        acc_vec[i,0,crt] = metrics.accuracy_score(Y_train, Y_pred_train)
        acc_vec[i,1,crt] = metrics.accuracy_score(Y_test, Y_pred_test)
        log_loss_vec[i,0,crt] = metrics.log_loss(Y_train, Y_pred_prob_train, normalize=True)
        log_loss_vec[i,1,crt] = metrics.log_loss(Y_test, Y_pred_prob_test, normalize=True)
        mis_clf_vec[i,0,crt] = np.sum(np.abs(Y_pred_train-Y_train.to_numpy()))/len(Y_train) * 100
        mis_clf_vec[i,1,crt] = np.sum(np.abs(Y_pred_test-Y_test.to_numpy()))/len(Y_test) * 100


#%%

print('... Results')
plt.rcdefaults()
plt.figure(figsize=(20,8))
plt.rc('font',size=18)
plt.subplot(1,2,1)
plt.plot(num_trees, acc_vec[:,0,0], '-o', lw=2, color='b', label='Gini/Train')
plt.plot(num_trees, acc_vec[:,1,0], '-o', lw=2, color='r', label='Gini/Test')
plt.plot(num_trees, acc_vec[:,0,1], '--s', lw=2, color='b', label='CE/Train')
plt.plot(num_trees, acc_vec[:,1,1], '--s', lw=2, color='r', label='CE/Test')
plt.xscale('log')
plt.legend(loc='best',fontsize=18)
plt.grid(axis='y',alpha=0.5)
plt.xticks(num_trees)
plt.xlabel('Num Trees')
plt.ylabel('Accuracy')
plt.subplot(1,2,2)
plt.plot(num_trees, log_loss_vec[:,0,0], '-o', lw=2, color='b', label='Gini/Train')
plt.plot(num_trees, log_loss_vec[:,1,0], '-o', lw=2, color='r', label='Gini/Test')
plt.plot(num_trees, log_loss_vec[:,0,1], '--s', lw=2, color='b', label='CE/Train')
plt.plot(num_trees, log_loss_vec[:,1,1], '--s', lw=2, color='r', label='CE/Test')
plt.xscale('log')
plt.legend(loc='best',fontsize=18)
plt.grid(axis='y',alpha=0.5)
plt.xticks(num_trees)
plt.xlabel('Num Trees')
plt.ylabel('Loss')

plt.figure(figsize=(10,8))
plt.rc('font',size=18)
plt.plot(num_trees, mis_clf_vec[:,0,0], '-o', lw=2, color='b', label='Gini/Train')
plt.plot(num_trees, mis_clf_vec[:,1,0], '-o', lw=2, color='r', label='Gini/Test')
plt.plot(num_trees, mis_clf_vec[:,0,1], '--s', lw=2, color='b', label='CE/Train')
plt.plot(num_trees, mis_clf_vec[:,1,1], '--s', lw=2, color='r', label='CE/Test')
plt.xscale('log')
plt.legend(loc='best',fontsize=18)
plt.grid(axis='y',alpha=0.5)
plt.xticks(num_trees)
plt.xlabel('Num Trees')
plt.ylabel('Misclassification Percent')
plt.show()

#%%

if True:
    print('\n> Applying Random Forest Classifier')
    tic = time.time()
    max_depth = 10
    num_trees = 1000
    rf_clf = RandomForestClassifier(n_estimators=num_trees, criterion='gini', max_depth=max_depth, bootstrap=True, oob_score=True, random_state=5)
    rf_clf.fit(X_train, Y_train)
    Y_pred = rf_clf.predict(X_test)
    Y_pred_prob = rf_clf.predict_proba(X_test)
    mis_clf = np.sum(np.abs(Y_pred-Y_test.to_numpy()))/len(Y_test) * 100
    toc = time.time()
    rf_clf_time = toc - tic
    
    #Evaluation
    print('\n================================')
    print ("%-15s%-15s"%('Metric','Value'))
    print('--------------------------------')
    print("%-15s%-15f"%('Accuracy', metrics.accuracy_score(Y_test, Y_pred)))
    print("%-15s%-15f"%('Precision', metrics.precision_score(Y_test, Y_pred, average='weighted')))
    print("%-15s%-15f"%('Recall', metrics.recall_score(Y_test, Y_pred, average='weighted')))
    print("%-15s%-15f"%('Log-Loss', metrics.log_loss(Y_test, Y_pred_prob, normalize=True)))
    print("%-15s%f %%"%('Misclassified', mis_clf))
    print('--------------------------------')
    print("Model: Random Forest")
    print("Max Tree Depth: %i"%max_depth)
    print("Num Estimators: %i"%num_trees)
    print("Criterion: Gini")
    print("Processed in %.6f seconds"%(rf_clf_time))
    print('================================\n')
    print("> Classification Report:")
    print(metrics.classification_report(Y_test, Y_pred))
    
    print("> Confusion Matrix")
    rf_clf_map = metrics.confusion_matrix(Y_test, Y_pred)
    plt.figure(figsize=(10,10))
    plt.rc('font', size=16)
    sns.heatmap(rf_clf_map, square=True, annot=True, fmt='d', cbar='False')
    plt.xlabel('Predicted Label',fontsize=22)
    plt.ylabel('True Label',fontsize=22)
    plt.show()


#%% OOB

print('\n> Finding best num trees for Random Forest Classifier')
num_trees = [10, 100, 500, 1000, 5000, 10000]
#num_trees = [10, 20, 30, 40, 50]
max_depth = 10
print('... Num trees range: {} to {}'.format(np.min(num_trees),np.max(num_trees)))
print('... Max depth: %i'%max_depth)
loss_funcs = ['gini','entropy']
print('... Criterions: ',loss_funcs)
oob_vec = np.zeros((len(num_trees),2))
for crt in range(len(loss_funcs)):
    for i in range(len(num_trees)):
        print('... %s with num trees = %i'%(loss_funcs[crt],num_trees[i]))
        num_est = num_trees[i]
        rf_clf = RandomForestClassifier(n_estimators=num_est, criterion=loss_funcs[crt], max_depth=max_depth, bootstrap=True, oob_score=True, random_state=5)
        rf_clf.fit(X_train, Y_train)
        oob_vec[i,crt] = rf_clf.oob_score_

#%% Plot OOB

plt.figure(figsize=(20,8))
plt.rc('font',size=18)
plt.subplot(1,2,1)
plt.plot(num_trees, oob_vec[:,0], '-o', lw=2, color='b', label='Gini')
plt.plot(num_trees, oob_vec[:,1], '--s', lw=2, color='b', label='CE')
plt.xscale('log')
plt.legend(loc='best',fontsize=18)
plt.grid(axis='y',alpha=0.5)
plt.xticks(num_trees)
plt.xlabel('Num Trees')
plt.ylabel('Out of Bag Score')
plt.subplot(1,2,2)
plt.plot(num_trees, 1-oob_vec[:,0], '-o', lw=2, color='r', label='Gini')
plt.plot(num_trees, 1-oob_vec[:,1], '--s', lw=2, color='r', label='CE')
plt.xscale('log')
plt.legend(loc='best',fontsize=18)
plt.grid(axis='y',alpha=0.5)
plt.xticks(num_trees)
plt.xlabel('Num Trees')
plt.ylabel('Out of Bag Error')
plt.show()


#%% Overfit - tree depth

print('\n> Trying to Overfit Random Forest Classifier')
num_trees = 100
max_depth_vec = [5, 10, 15, 20, 25, 30]
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
        rf_clf = RandomForestClassifier(n_estimators=num_trees, criterion=loss_funcs[crt], max_depth=max_depth, bootstrap=True, oob_score=True, random_state=5)
        rf_clf.fit(X_train, Y_train)
        Y_pred_train = rf_clf.predict(X_train)
        Y_pred_test = rf_clf.predict(X_test)
        acc_vec[i,0,crt] = metrics.accuracy_score(Y_train, Y_pred_train)
        acc_vec[i,1,crt] = metrics.accuracy_score(Y_test, Y_pred_test)
        mis_clf_vec[i,0,crt] = np.sum(np.abs(Y_pred_train-Y_train.to_numpy()))/len(Y_train)
        mis_clf_vec[i,1,crt] = np.sum(np.abs(Y_pred_test-Y_test.to_numpy()))/len(Y_test)
        

#%% Result - tree depth

print('... Results')
plt.rcdefaults()
plt.figure(figsize=(20,8))
plt.rc('font',size=18)
plt.subplot(1,2,1)
plt.plot(max_depth_vec, acc_vec[:,0,0], '-o', lw=2, color='b', label='Gini/Train')
plt.plot(max_depth_vec, acc_vec[:,1,0], '-o', lw=2, color='r', label='Gini/Test')
plt.plot(max_depth_vec, acc_vec[:,0,1], '--s', lw=2, color='b', label='CE/Train')
plt.plot(max_depth_vec, acc_vec[:,1,1], '--s', lw=2, color='r', label='CE/Test')
plt.legend(loc='upper left',fontsize=18)
plt.grid(axis='y',alpha=0.5)
plt.xticks(max_depth_vec)
plt.xlabel('Max Tree Depth')
plt.ylabel('Accuracy')
plt.subplot(1,2,2)
plt.plot(max_depth_vec, mis_clf_vec[:,0,0], '-o', lw=2, color='b', label='Gini/Train')
plt.plot(max_depth_vec, mis_clf_vec[:,1,0], '-o', lw=2, color='r', label='Gini/Test')
plt.plot(max_depth_vec, mis_clf_vec[:,0,1], '--s', lw=2, color='b', label='CE/Train')
plt.plot(max_depth_vec, mis_clf_vec[:,1,1], '--s', lw=2, color='r', label='CE/Test')
plt.legend(loc='lower left',fontsize=18)
plt.grid(axis='y',alpha=0.5)
plt.xticks(max_depth_vec)
plt.xlabel('Max Tree Depth')
plt.ylabel('Misclassification Error')
plt.suptitle('Random Forest Train-Test Accuracy / Error')
plt.show()


#%% Overfit - max feature

print('\n> Trying to Overfit Random Forest Classifier')
num_trees = 100
max_depth = 10
max_features_vec = [1, 5, 10, 20, 30, 50, 80]
print('... Num trees: %i'%num_trees)
print('... Max depth: %i'%max_depth)
print('... Max features range: {} to {}'.format(np.min(max_features_vec),np.max(max_features_vec)))
loss_funcs = ['gini','entropy']
print('... Criterions: ',loss_funcs)
acc_vec = np.zeros((len(max_features_vec),2,2),dtype=float)
mis_clf_vec = np.zeros((len(max_features_vec),2,2),dtype=float)
for crt in range(len(loss_funcs)):
    for i in range(len(max_features_vec)):
        print('... %s with max features = %i'%(loss_funcs[crt],max_features_vec[i]))
        max_features = max_features_vec[i]
        rf_clf = RandomForestClassifier(n_estimators=num_trees, criterion=loss_funcs[crt], max_depth=max_depth, bootstrap=True,
                                        oob_score=True, random_state=5, max_features=max_features)
        rf_clf.fit(X_train, Y_train)
        Y_pred_train = rf_clf.predict(X_train)
        Y_pred_test = rf_clf.predict(X_test)
        acc_vec[i,0,crt] = metrics.accuracy_score(Y_train, Y_pred_train)
        acc_vec[i,1,crt] = metrics.accuracy_score(Y_test, Y_pred_test)
        mis_clf_vec[i,0,crt] = np.sum(np.abs(Y_pred_train-Y_train.to_numpy()))/len(Y_train)
        mis_clf_vec[i,1,crt] = np.sum(np.abs(Y_pred_test-Y_test.to_numpy()))/len(Y_test)


#%% Result - max features

print('... Results')
plt.rcdefaults()
plt.figure(figsize=(20,8))
plt.rc('font',size=18)
plt.subplot(1,2,1)
plt.plot(max_features_vec, acc_vec[:,0,0], '-o', lw=2, color='b', label='Gini/Train')
plt.plot(max_features_vec, acc_vec[:,1,0], '-o', lw=2, color='r', label='Gini/Test')
plt.plot(max_features_vec, acc_vec[:,0,1], '--s', lw=2, color='b', label='CE/Train')
plt.plot(max_features_vec, acc_vec[:,1,1], '--s', lw=2, color='r', label='CE/Test')
plt.legend(loc='lower right',fontsize=18)
plt.grid(axis='y',alpha=0.5)
plt.xticks(max_features_vec)
plt.xlabel('Max Features')
plt.ylabel('Accuracy')
plt.subplot(1,2,2)
plt.plot(max_features_vec, mis_clf_vec[:,0,0], '-o', lw=2, color='b', label='Gini/Train')
plt.plot(max_features_vec, mis_clf_vec[:,1,0], '-o', lw=2, color='r', label='Gini/Test')
plt.plot(max_features_vec, mis_clf_vec[:,0,1], '--s', lw=2, color='b', label='CE/Train')
plt.plot(max_features_vec, mis_clf_vec[:,1,1], '--s', lw=2, color='r', label='CE/Test')
plt.legend(loc='upper right',fontsize=18)
plt.grid(axis='y',alpha=0.5)
plt.xticks(max_features_vec)
plt.xlabel('Max Features')
plt.ylabel('Misclassification Error')
plt.suptitle('Random Forest Train-Test Accuracy / Error')
plt.show()


#%% Grid Search CV

print('\n> Grid Search Cross-Validation for Bagging')
tic = time.time()
model_cv = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=1, bootstrap=True,
                                oob_score=True, max_features=20)
n_trees = [100, 500, 1000, 5000]
loss_funcs = ['gini', 'entropy']
max_depth = [1, 3, 5, 10]
max_features = [20, 40, 80]
param_grid = dict(n_estimators=n_trees, criterion=loss_funcs, max_depth=max_depth, max_features=max_features)
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

