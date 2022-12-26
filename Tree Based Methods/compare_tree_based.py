# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 12:13:01 2022

@author: mz52
"""

from tree_methods_lib import *
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import time
import warnings
warnings.filterwarnings('ignore')

# %% Preprocessing

# load train and test datasets
data_train, data_test, features = load_dataset(print_info=False)
# prepare datasets for classification
X_train, X_test, Y_train, Y_test = preparing_dataset(data_train, data_test)

#%% Decision Tree

print('\n> Decision Tree Classifier')
tic = time.time()
dt_clf = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=9, random_state=5)
dt_clf.fit(X_train, Y_train)
dt_feature_importance = np.flip(np.argsort(dt_clf.feature_importances_))
Y_pred_train = dt_clf.predict(X_train)
Y_pred_test = dt_clf.predict(X_test)
Y_pred_prob_train = dt_clf.predict_proba(X_train)
Y_pred_prob_test = dt_clf.predict_proba(X_test)
dt_acc_train = metrics.accuracy_score(Y_train, Y_pred_train)
dt_acc_test = metrics.accuracy_score(Y_test, Y_pred_test)
dt_recall_train = metrics.recall_score(Y_train, Y_pred_train, average='weighted')
dt_recall_test = metrics.recall_score(Y_test, Y_pred_test, average='weighted')
dt_log_loss_train = metrics.log_loss(Y_train, Y_pred_prob_train, normalize=True)
dt_log_loss_test = metrics.log_loss(Y_test, Y_pred_prob_test, normalize=True)
dt_mis_clf_train = np.sum(np.abs(Y_pred_train-Y_train.to_numpy()))/len(Y_train)
dt_mis_clf_test = np.sum(np.abs(Y_pred_test-Y_test.to_numpy()))/len(Y_test)
dt_conf_mat = metrics.confusion_matrix(Y_test, Y_pred_test)
toc = time.time()
dt_time = toc - tic
print('... done!')
print('... processing time: %.6f'%dt_time)

#%% Bagging

print('\n> Bagging Classifier')
tic = time.time()
base_clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=3)
bg_clf = BaggingClassifier(base_estimator=base_clf, n_estimators=1000, bootstrap=True, oob_score=True, random_state=5)
bg_clf.fit(X_train, Y_train)
bg_feature_importance = np.flip(np.argsort(np.mean([tree.feature_importances_ for tree in bg_clf.estimators_], axis=0)))
Y_pred_train = bg_clf.predict(X_train)
Y_pred_test = bg_clf.predict(X_test)
Y_pred_prob_train = bg_clf.predict_proba(X_train)
Y_pred_prob_test = bg_clf.predict_proba(X_test)
bg_acc_train = metrics.accuracy_score(Y_train, Y_pred_train)
bg_acc_test = metrics.accuracy_score(Y_test, Y_pred_test)
bg_recall_train = metrics.recall_score(Y_train, Y_pred_train, average='weighted')
bg_recall_test = metrics.recall_score(Y_test, Y_pred_test, average='weighted')
bg_log_loss_train = metrics.log_loss(Y_train, Y_pred_prob_train, normalize=True)
bg_log_loss_test = metrics.log_loss(Y_test, Y_pred_prob_test, normalize=True)
bg_mis_clf_train = np.sum(np.abs(Y_pred_train-Y_train.to_numpy()))/len(Y_train)
bg_mis_clf_test = np.sum(np.abs(Y_pred_test-Y_test.to_numpy()))/len(Y_test)
bg_conf_mat = metrics.confusion_matrix(Y_test, Y_pred_test)
toc = time.time()
bg_time = toc - tic
print('... done!')
print('... processing time: %.6f'%bg_time)

#%% Random Forest

print('\n> Random Forest Classifier')
tic = time.time()
rf_clf = RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=10, bootstrap=True,
                                oob_score=True, max_features=None, random_state=5)
rf_clf.fit(X_train, Y_train)
rf_feature_importance = np.flip(np.argsort(rf_clf.feature_importances_))
Y_pred_train = rf_clf.predict(X_train)
Y_pred_test = rf_clf.predict(X_test)
Y_pred_prob_train = rf_clf.predict_proba(X_train)
Y_pred_prob_test = rf_clf.predict_proba(X_test)
rf_acc_train = metrics.accuracy_score(Y_train, Y_pred_train)
rf_acc_test = metrics.accuracy_score(Y_test, Y_pred_test)
rf_recall_train = metrics.recall_score(Y_train, Y_pred_train, average='weighted')
rf_recall_test = metrics.recall_score(Y_test, Y_pred_test, average='weighted')
rf_log_loss_train = metrics.log_loss(Y_train, Y_pred_prob_train, normalize=True)
rf_log_loss_test = metrics.log_loss(Y_test, Y_pred_prob_test, normalize=True)
rf_mis_clf_train = np.sum(np.abs(Y_pred_train-Y_train.to_numpy()))/len(Y_train)
rf_mis_clf_test = np.sum(np.abs(Y_pred_test-Y_test.to_numpy()))/len(Y_test)
rf_conf_mat = metrics.confusion_matrix(Y_test, Y_pred_test)
toc = time.time()
rf_time = toc - tic
print('... done!')
print('... processing time: %.6f'%rf_time)

#%% AdaBoost

print('\n> AdaBoost Classifier')
tic = time.time()
base_clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=3)
ab_clf = AdaBoostClassifier(base_estimator=base_clf, n_estimators=500, learning_rate=0.5, random_state=5)
ab_clf.fit(X_train, Y_train)
ab_feature_importance = np.flip(np.argsort(ab_clf.feature_importances_))
Y_pred_train = ab_clf.predict(X_train)
Y_pred_test = ab_clf.predict(X_test)
Y_pred_prob_train = ab_clf.predict_proba(X_train)
Y_pred_prob_test = ab_clf.predict_proba(X_test)
ab_acc_train = metrics.accuracy_score(Y_train, Y_pred_train)
ab_acc_test = metrics.accuracy_score(Y_test, Y_pred_test)
ab_recall_train = metrics.recall_score(Y_train, Y_pred_train, average='weighted')
ab_recall_test = metrics.recall_score(Y_test, Y_pred_test, average='weighted')
ab_log_loss_train = metrics.log_loss(Y_train, Y_pred_prob_train, normalize=True)
ab_log_loss_test = metrics.log_loss(Y_test, Y_pred_prob_test, normalize=True)
ab_mis_clf_train = np.sum(np.abs(Y_pred_train-Y_train.to_numpy()))/len(Y_train)
ab_mis_clf_test = np.sum(np.abs(Y_pred_test-Y_test.to_numpy()))/len(Y_test)
ab_conf_mat = metrics.confusion_matrix(Y_test, Y_pred_test)
toc = time.time()
ab_time = toc - tic
print('... done!')
print('... processing time: %.6f'%ab_time)

#%% Gradient Boosting

print('\n> Gradient Boosting Classifier')
tic = time.time()
gb_clf = GradientBoostingClassifier(loss='exponential', learning_rate=0.05, n_estimators=500,
                                    max_depth=5, max_features=None, random_state=5)
gb_clf.fit(X_train, Y_train)
gb_feature_importance = np.flip(np.argsort(gb_clf.feature_importances_))
Y_pred_train = gb_clf.predict(X_train)
Y_pred_test = gb_clf.predict(X_test)
Y_pred_prob_train = gb_clf.predict_proba(X_train)
Y_pred_prob_test = gb_clf.predict_proba(X_test)
gb_acc_train = metrics.accuracy_score(Y_train, Y_pred_train)
gb_acc_test = metrics.accuracy_score(Y_test, Y_pred_test)
gb_recall_train = metrics.recall_score(Y_train, Y_pred_train, average='weighted')
gb_recall_test = metrics.recall_score(Y_test, Y_pred_test, average='weighted')
gb_log_loss_train = metrics.log_loss(Y_train, Y_pred_prob_train, normalize=True)
gb_log_loss_test = metrics.log_loss(Y_test, Y_pred_prob_test, normalize=True)
gb_mis_clf_train = np.sum(np.abs(Y_pred_train-Y_train.to_numpy()))/len(Y_train)
gb_mis_clf_test = np.sum(np.abs(Y_pred_test-Y_test.to_numpy()))/len(Y_test)
gb_conf_mat = metrics.confusion_matrix(Y_test, Y_pred_test)
toc = time.time()
gb_time = toc - tic
print('... done!')
print('... processing time: %.6f'%gb_time)


#%% Results

#Evaluation
print ('\n%-25s%-10s'%(' ','** Performance of Tree-Based Models **'))
print('===========================================================================================')
print ("%-20s%-15s%-15s%-15s%-15s%-15s"%('Metric','Tree','Bagging','RF','AdaBoost','GB'))
print('===========================================================================================')
print("%-20s%-15.4f%-15.4f%-15.4f%-15.4f%-15.4f"%('Train Accuracy', dt_acc_train, bg_acc_train, rf_acc_train, ab_acc_train, gb_acc_train))
print("%-20s%-15.4f%-15.4f%-15.4f%-15.4f%-15.4f"%('Test Accuracy', dt_acc_test, bg_acc_test, rf_acc_test, ab_acc_test, gb_acc_test))
print('-------------------------------------------------------------------------------------------')
print("%-20s%-15.4f%-15.4f%-15.4f%-15.4f%-15.4f"%('Train Recall', dt_recall_train, bg_recall_train, rf_recall_train, ab_recall_train, gb_recall_train))
print("%-20s%-15.4f%-15.4f%-15.4f%-15.4f%-15.4f"%('Test Recall', dt_recall_test, bg_recall_test, rf_recall_test, ab_recall_test, gb_recall_test))
print('-------------------------------------------------------------------------------------------')
print("%-20s%-15.4f%-15.4f%-15.4f%-15.4f%-15.4f"%('Train Log-Loss', dt_log_loss_train, bg_log_loss_train, rf_log_loss_train, ab_log_loss_train, gb_log_loss_train))
print("%-20s%-15.4f%-15.4f%-15.4f%-15.4f%-15.4f"%('Test Log-Loss', dt_log_loss_test, bg_log_loss_test, rf_log_loss_test, ab_log_loss_test, gb_log_loss_test))
print('-------------------------------------------------------------------------------------------')
print("%-20s%-15.4f%-15.4f%-15.4f%-15.4f%-15.4f"%('Train Mis-clf', dt_mis_clf_train, bg_mis_clf_train, rf_mis_clf_train, ab_mis_clf_train, gb_mis_clf_train))
print("%-20s%-15.4f%-15.4f%-15.4f%-15.4f%-15.4f"%('Test Mis-clf', dt_mis_clf_test, bg_mis_clf_test, rf_mis_clf_test, ab_mis_clf_test, gb_mis_clf_test))
print('-------------------------------------------------------------------------------------------')
print("%-20s%-15.6f%-15.6f%-15.6f%-15.6f%-15.6f"%('Procc Time', dt_time, bg_time, rf_time, ab_time, gb_time))
print('===========================================================================================')

#%% Conf Matrix

print("\n> Confusion Matrix")
plt.figure(figsize=(10,10))
plt.rc('font', size=16)
plt.title('Decision Tree')
sns.heatmap(dt_conf_mat, square=True, annot=True, fmt='d', cbar='False')
plt.xlabel('Predicted Label',fontsize=22)
plt.ylabel('True Label',fontsize=22)

plt.figure(figsize=(10,10))
plt.rc('font', size=16)
plt.title('Bagging')
sns.heatmap(bg_conf_mat, square=True, annot=True, fmt='d', cbar='False')
plt.xlabel('Predicted Label',fontsize=22)
plt.ylabel('True Label',fontsize=22)

plt.figure(figsize=(10,10))
plt.rc('font', size=16)
plt.title('Random Forest')
sns.heatmap(rf_conf_mat, square=True, annot=True, fmt='d', cbar='False')
plt.xlabel('Predicted Label',fontsize=22)
plt.ylabel('True Label',fontsize=22)

plt.figure(figsize=(10,10))
plt.rc('font', size=16)
plt.title('AdaBoost')
sns.heatmap(ab_conf_mat, square=True, annot=True, fmt='d', cbar='False')
plt.xlabel('Predicted Label',fontsize=22)
plt.ylabel('True Label',fontsize=22)

plt.figure(figsize=(10,10))
plt.rc('font', size=16)
plt.title('Gradient Boosting')
sns.heatmap(gb_conf_mat, square=True, annot=True, fmt='d', cbar='False')
plt.xlabel('Predicted Label',fontsize=22)
plt.ylabel('True Label',fontsize=22)

plt.show()

#%% Feature importance

feature_names = X_train.columns

print ('\n%-65s%-10s'%(' ','** Feature Importance for Tree-Based Models **'))
print('==========================================================================================='
      '=================================================================================')
print ("%-35s%-35s%-35s%-35s%-35s"%('Tree','Bagging','RF','AdaBoost','GB'))
print('==========================================================================================='
      '=================================================================================')
for i in range(25):
    print("%-35s%-35s%-35s%-35s%-35s"%(feature_names[dt_feature_importance[i]], feature_names[bg_feature_importance[i]],
                                       feature_names[rf_feature_importance[i]], feature_names[ab_feature_importance[i]], feature_names[gb_feature_importance[i]]))
print('==========================================================================================='
      '=================================================================================')








