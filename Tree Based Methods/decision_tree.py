# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 14:42:14 2022

@author: mz52
"""

from tree_methods_lib import *
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.ensemble import BaggingClassifier
import time

# %% Preprocessing

# load train and test datasets
data_train, data_test, features = load_dataset(print_info=False)
# prepare datasets for classification
X_train, X_test, Y_train, Y_test = preparing_dataset(data_train, data_test)

#%% Decision Tree Classifier

print('\n> Finding best depth size for Decision Tree Classifier')
tree_depths = np.arange(2,21)
print('... Depth range: {} to {}'.format(np.min(tree_depths),np.max(tree_depths)))
loss_funcs = ['gini','entropy']
print('... Criterions: ',loss_funcs)
acc_vec = np.zeros((len(tree_depths),2,2),dtype=float)
log_loss_vec = np.zeros((len(tree_depths),2,2),dtype=float)
mis_clf_vec = np.zeros((len(tree_depths),2,2),dtype=float)
for crt in range(len(loss_funcs)):
    for i in range(len(tree_depths)):
        best_depth = tree_depths[i]
        dt_clf = DecisionTreeClassifier(criterion=loss_funcs[crt], splitter='best', max_depth=best_depth)
        dt_clf.fit(X_train, Y_train)
        Y_pred_train = dt_clf.predict(X_train)
        Y_pred_test = dt_clf.predict(X_test)
        Y_pred_prob_train = dt_clf.predict_proba(X_train)
        Y_pred_prob_test = dt_clf.predict_proba(X_test)
        acc_vec[i,0,crt] = metrics.accuracy_score(Y_train, Y_pred_train)
        acc_vec[i,1,crt] = metrics.accuracy_score(Y_test, Y_pred_test)
        log_loss_vec[i,0,crt] = metrics.log_loss(Y_train, Y_pred_prob_train, normalize=True)
        log_loss_vec[i,1,crt] = metrics.log_loss(Y_test, Y_pred_prob_test, normalize=True)
        mis_clf_vec[i,0,crt] = np.sum(np.abs(Y_pred_train-Y_train.to_numpy()))/len(Y_train) * 100
        mis_clf_vec[i,1,crt] = np.sum(np.abs(Y_pred_test-Y_test.to_numpy()))/len(Y_test) * 100


#%% Showing results

print('... Results')
plt.rcdefaults()
plt.figure(figsize=(20,8))
plt.rc('font',size=18)
plt.subplot(1,2,1)
plt.plot(tree_depths, acc_vec[:,0,0], '-o', lw=2, color='b', label='Gini/Train')
plt.plot(tree_depths, acc_vec[:,1,0], '-o', lw=2, color='r', label='Gini/Test')
plt.plot(tree_depths, acc_vec[:,0,1], '--s', lw=2, color='b', label='CE/Train')
plt.plot(tree_depths, acc_vec[:,1,1], '--s', lw=2, color='r', label='CE/Test')
plt.legend(loc='upper left',fontsize=18)
plt.grid(axis='y',alpha=0.5)
plt.xticks(tree_depths)
plt.xlabel('Max Tree Depth')
plt.ylabel('Accuracy')
plt.subplot(1,2,2)
plt.plot(tree_depths, log_loss_vec[:,0,0], '-o', lw=2, color='b', label='Gini/Train')
plt.plot(tree_depths, log_loss_vec[:,1,0], '-o', lw=2, color='r', label='Gini/Test')
plt.plot(tree_depths, log_loss_vec[:,0,1], '--s', lw=2, color='b', label='CE/Train')
plt.plot(tree_depths, log_loss_vec[:,1,1], '--s', lw=2, color='r', label='CE/Test')
plt.legend(loc='upper left',fontsize=18)
plt.grid(axis='y',alpha=0.5)
plt.xticks(tree_depths)
plt.xlabel('Max Tree Depth')
plt.ylabel('Loss')

plt.figure(figsize=(10,8))
plt.rc('font',size=18)
plt.plot(tree_depths, mis_clf_vec[:,0,0], '-o', lw=2, color='b', label='Gini/Train')
plt.plot(tree_depths, mis_clf_vec[:,1,0], '-o', lw=2, color='r', label='Gini/Test')
plt.plot(tree_depths, mis_clf_vec[:,0,1], '--s', lw=2, color='b', label='CE/Train')
plt.plot(tree_depths, mis_clf_vec[:,1,1], '--s', lw=2, color='r', label='CE/Test')
plt.legend(loc='lower left',fontsize=18)
plt.grid(axis='y',alpha=0.5)
plt.xticks(tree_depths)
plt.xlabel('Max Tree Depth')
plt.ylabel('Misclassification Percent')
plt.show()

#%% Best Depth Size

print('\n> Applying DTC with the best tree size')
best_depth = 10
tic = time.time()
dt_clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=best_depth)
dt_clf.fit(X_train, Y_train)
Y_pred = dt_clf.predict(X_test)
Y_pred_prob = dt_clf.predict_proba(X_test)
mis_clf = np.sum(np.abs(Y_pred-Y_test.to_numpy()))/len(Y_test) * 100
toc = time.time()
dt_clf_time = toc - tic

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
print("Model: Decision Tree")
print("Criterion: Gini")
print("Max depth: {}".format(best_depth))
print("Processed in %.6f seconds"%(dt_clf_time))
print('================================\n')
print("> Classification Report:")
print(metrics.classification_report(Y_test, Y_pred))

print("> Confusion Matrix")
dt_clf_mat = metrics.confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(10,10))
plt.rc('font', size=16)
sns.heatmap(dt_clf_mat, square=True, annot=True, fmt='d', cbar='False')
plt.xlabel('Predicted Label',fontsize=22)
plt.ylabel('True Label',fontsize=22)
plt.show()


#%% Plot Tree

labels = ['0','1']
#create the tree plot
plt.figure(figsize=(20,10))
a = plot_tree(dt_clf, feature_names = X_train.columns, class_names = labels, label='root', max_depth=4,
              rounded = True, filled = True, fontsize=5)
#show the plot
plt.show()


#%% Overfit


print('\n> Trying to Overfit Decision Tree Classifier')
tree_depths = np.arange(2,26)
print('... Depth range: {} to {}'.format(np.min(tree_depths),np.max(tree_depths)))
loss_funcs = ['gini','entropy']
print('... Criterions: ',loss_funcs)
acc_vec = np.zeros((len(tree_depths),2,2),dtype=float)
mis_clf_vec = np.zeros((len(tree_depths),2,2),dtype=float)
for crt in range(len(loss_funcs)):
    for i in range(len(tree_depths)):
        best_depth = tree_depths[i]
        dt_clf = DecisionTreeClassifier(criterion=loss_funcs[crt], splitter='best', max_depth=best_depth)
        dt_clf.fit(X_train, Y_train)
        Y_pred_train = dt_clf.predict(X_train)
        Y_pred_test = dt_clf.predict(X_test)
        Y_pred_prob_train = dt_clf.predict_proba(X_train)
        Y_pred_prob_test = dt_clf.predict_proba(X_test)
        acc_vec[i,0,crt] = metrics.accuracy_score(Y_train, Y_pred_train)
        acc_vec[i,1,crt] = metrics.accuracy_score(Y_test, Y_pred_test)
        mis_clf_vec[i,0,crt] = np.sum(np.abs(Y_pred_train-Y_train.to_numpy()))/len(Y_train)
        mis_clf_vec[i,1,crt] = np.sum(np.abs(Y_pred_test-Y_test.to_numpy()))/len(Y_test)


#%% Results

print('... Results')
plt.rcdefaults()
plt.figure(figsize=(20,8))
plt.rc('font',size=18)
plt.subplot(1,2,1)
plt.plot(tree_depths, acc_vec[:,0,0], '-o', lw=2, color='b', label='Gini/Train')
plt.plot(tree_depths, acc_vec[:,1,0], '-o', lw=2, color='r', label='Gini/Test')
plt.plot(tree_depths, acc_vec[:,0,1], '--s', lw=2, color='b', label='CE/Train')
plt.plot(tree_depths, acc_vec[:,1,1], '--s', lw=2, color='r', label='CE/Test')
plt.legend(loc='upper left',fontsize=18)
plt.grid(axis='y',alpha=0.5)
plt.xticks(np.arange(2,26,2))
plt.xlabel('Max Tree Depth')
plt.ylabel('Accuracy')
plt.subplot(1,2,2)
plt.plot(tree_depths, mis_clf_vec[:,0,0], '-o', lw=2, color='b', label='Gini/Train')
plt.plot(tree_depths, mis_clf_vec[:,1,0], '-o', lw=2, color='r', label='Gini/Test')
plt.plot(tree_depths, mis_clf_vec[:,0,1], '--s', lw=2, color='b', label='CE/Train')
plt.plot(tree_depths, mis_clf_vec[:,1,1], '--s', lw=2, color='r', label='CE/Test')
plt.legend(loc='lower left',fontsize=18)
plt.grid(axis='y',alpha=0.5)
plt.xticks(np.arange(2,26,2))
plt.xlabel('Max Tree Depth')
plt.ylabel('Misclassification Error')
plt.suptitle('Tree Test Accuracy / Error')
plt.show()


#%% Grid Search CV

print('\n> Grid Search Cross-Validation for Decision Tree')
tic = time.time()
model_cv = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=5)
loss_funcs = ['gini', 'entropy']
max_depth = np.arange(1,27,2)
param_grid = dict(criterion=loss_funcs, max_depth=max_depth)
grid = GridSearchCV(estimator=model_cv, param_grid=param_grid, cv=3, verbose=3)
grid_result = grid.fit(X_train, Y_train)
grid_best_params = grid.best_params_
toc = time.time()
dt_grid_time = toc - tic

print("\nprocessing time: %.6f sec\n"%dt_grid_time)
print(f'Best Accuracy for {grid_result.best_score_:.4} using {grid_result.best_params_}')
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(f'mean={mean:.4}, std={stdev:.4} using {param}')



