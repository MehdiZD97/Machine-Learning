# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 03:20:38 2022

@author: mz52
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests, gzip, os, hashlib
from sklearn import svm, metrics
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import discriminant_analysis
from sklearn import multiclass
import time
import warnings
from sklearn.model_selection import GridSearchCV

#%% Loading The Dataset

scale_data = True
flatten_data = True

# fetch data
#path = 'C:/Users/mz52/MATLAB-Drive/2-1 Fall 2022/ELEC 578/Dataset'
path = 'Dataset'
def fetch(url):
    fp = os.path.join(path, hashlib.md5(url.encode('utf-8')).hexdigest())
    if os.path.isfile(fp):
        with open(fp, "rb") as f:
            data = f.read()
    else:
        with open(fp, "wb") as f:
            data = requests.get(url).content
            f.write(data)
    return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()

X_train_mat = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28,28))
Y_train = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
X_test_mat = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28,28))
Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]

# scaling
X_train = (X_train_mat/255) if scale_data else X_train_mat
X_test = (X_test_mat/255) if scale_data else X_test_mat
# flattenning
X_train = X_train.reshape(len(X_train),28*28) if flatten_data else X_train
X_test = X_test.reshape(len(X_test),28*28) if flatten_data else X_test

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%% (1) Logistic Multinomial Regression

# Grid search for LR
grid_lr_activate = False
if grid_lr_activate:
    print("\n** Grid Search for Logistic Regression **\n")
    tic = time.time()
    print("Grid search ...")
    param_grid = {'C': 10**np.arange(-2,1.2,0.2,dtype=float)} 
    grid = GridSearchCV(linear_model.LogisticRegression(penalty='l2',max_iter=200,multi_class='multinomial'), param_grid, refit = True, verbose = 3)
    grid.fit(X_train, Y_train)
    grid_best_lr = grid.best_params_
    toc = time.time()
    lr_grid_time = toc - tic
    print("done!")
    print("processing time: %.6f sec"%lr_grid_time)
    print("Best parameters found by grid search:")
    print(grid.best_params_)

#%% LR with Best Params

LR_activate = False
if LR_activate:
    print("\n** Logistic Multinomial Regression **\n")
    tic = time.time()
    best_C_lr = grid_best_lr['C']
    lr_clf = linear_model.LogisticRegression(penalty='l2',C=best_C_lr,max_iter=500,multi_class='multinomial')
    lr_clf.fit(X_train, Y_train)
    Y_predict = lr_clf.predict(X_test)
    Y_predict_prob = lr_clf.predict_proba(X_test)
    toc = time.time()
    lr_time = toc - tic
    
    #Evaluation
    print('\n================================')
    print ("%-15s%-15s"%('Metric','Value'))
    print('--------------------------------')
    print("%-15s%-15f"%('Accuracy', metrics.accuracy_score(Y_test, Y_predict)))
    print("%-15s%-15f"%('Precision', metrics.precision_score(Y_test, Y_predict, average='weighted')))
    print("%-15s%-15f"%('Recall', metrics.recall_score(Y_test, Y_predict, average='weighted')))
    print("%-15s%-15f"%('Log-Loss', metrics.log_loss(Y_test, Y_predict_prob, normalize=True)))
    print('--------------------------------')
    print("Model: Logistic Multinomial Regression")
    print("C value: {}".format(best_C_lr))
    print("Processed in %.6f seconds"%(lr_time))
    print('================================\n')
    print("Classification Report:")
    print(metrics.classification_report(Y_test, Y_predict))

    print("Confusion Matrix")
    lr_mat = metrics.confusion_matrix(Y_test, Y_predict)
    plt.figure(figsize=(12,12))
    plt.rc('font', size=20)
    sns.heatmap(lr_mat, square=True, annot=True, fmt='d', cbar='False')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    print("Pixel Heatmap")
    plt.figure(figsize=(25,10))
    #plt.rc('font', size=14)
    for i in range(10):
        plt.subplot(2,5,i+1)
        lr_coef = np.abs(np.reshape(lr_clf.coef_[i,:].flatten(),(28,28)))
        sns.heatmap(lr_coef,square=True)
        plt.axis('off')
        plt.title('Class {}'.format(i))
    plt.show()

#%% (2) Naive Bayes Classifier

gnb_activate = False
if gnb_activate:
    print("\n** Gaussian Naive Bayes **\n")
    tic = time.time()
    gnb_clf = naive_bayes.GaussianNB()
    gnb_clf.fit(X_train, Y_train)
    Y_predict = gnb_clf.predict(X_test)
    Y_predict_prob = gnb_clf.predict_proba(X_test)
    toc = time.time()
    gnb_time = toc - tic
    
    #Evaluation
    print('\n================================')
    print ("%-15s%-15s"%('Metric','Value'))
    print('--------------------------------')
    print("%-15s%-15f"%('Accuracy', metrics.accuracy_score(Y_test, Y_predict)))
    print("%-15s%-15f"%('Precision', metrics.precision_score(Y_test, Y_predict, average='weighted')))
    print("%-15s%-15f"%('Recall', metrics.recall_score(Y_test, Y_predict, average='weighted')))
    print("%-15s%-15f"%('Log-Loss', metrics.log_loss(Y_test, Y_predict_prob, normalize=True)))
    print('--------------------------------')
    print("Model: Gaussian Naive Bayes")
    print("Processed in %.6f seconds"%(gnb_time))
    print('================================\n')
    print("Classification Report:")
    print(metrics.classification_report(Y_test, Y_predict))
    
    print("Confusion Matrix")
    gnb_mat = metrics.confusion_matrix(Y_test, Y_predict)
    plt.figure(figsize=(12,12))
    plt.rc('font', size=20)
    sns.heatmap(gnb_mat, square=True, annot=True, fmt='d', cbar='False')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

#%% (3) Linear Discriminant Analysis

lda_activate = False
if lda_activate:
    print("\n** Linear Discriminant Analysis **\n")
    tic = time.time()
    lda_clf = discriminant_analysis.LinearDiscriminantAnalysis()
    lda_clf.fit(X_train, Y_train)
    Y_predict = lda_clf.predict(X_test)
    Y_predict_prob = lda_clf.predict_proba(X_test)
    toc = time.time()
    lda_time = toc - tic
    
    #Evaluation
    print('\n================================')
    print ("%-15s%-15s"%('Metric','Value'))
    print('--------------------------------')
    print("%-15s%-15f"%('Accuracy', metrics.accuracy_score(Y_test, Y_predict)))
    print("%-15s%-15f"%('Precision', metrics.precision_score(Y_test, Y_predict, average='weighted')))
    print("%-15s%-15f"%('Recall', metrics.recall_score(Y_test, Y_predict, average='weighted')))
    print("%-15s%-15f"%('Log-Loss', metrics.log_loss(Y_test, Y_predict_prob, normalize=True)))
    print('--------------------------------')
    print("Model: Linear Discriminant Analysis")
    print("Processed in %.6f seconds"%(lda_time))
    print('================================\n')
    print("Classification Report:")
    print(metrics.classification_report(Y_test, Y_predict))
    
    print("Confusion Matrix")
    lda_mat = metrics.confusion_matrix(Y_test, Y_predict)
    plt.figure(figsize=(12,12))
    plt.rc('font', size=20)
    sns.heatmap(lda_mat, square=True, annot=True, fmt='d', cbar='False')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    
    print("Pixel Heatmap")
    plt.figure(figsize=(25,10))
    #plt.rc('font', size=14)
    for i in range(10):
        plt.subplot(2,5,i+1)
        lda_coef = np.abs(np.reshape(lda_clf.coef_[i,:].flatten(),(28,28)))
        sns.heatmap(lda_coef,square=True)
        plt.axis('off')
        plt.title('Class {}'.format(i))
    plt.show()

#%% (4-1) Linear SVM OVO

grid_ovo_activate = False
if grid_ovo_activate:
    print("\n** Grid Search for Linear SVM **\n")
    tic = time.time()
    print("Grid search ...")
    param_grid = {'C': 10**np.arange(-1.5,0.9,0.4,dtype=float)} 
    grid = GridSearchCV(svm.LinearSVC(penalty='l2', multiclass='ovo'), param_grid, refit = True, verbose = 3)
    grid.fit(X_train, Y_train)
    grid_best_svm = grid.best_params_
    toc = time.time()
    svm_grid_time = toc - tic
    print("done!")
    print("processing time: %.6f sec"%svm_grid_time)
    print("Best parameters found by grid search:")
    print(grid.best_params_)
    best_C_lsvm = grid_best_svm['C']
    

#%% SVM with OVO

svm_ovo_activate = False
if svm_ovo_activate:
    print("\n** Linear SVM One Vs. One **\n")
    tic = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        multiclass_ovo_clf = multiclass.OneVsOneClassifier(svm.LinearSVC(penalty='l2', C=best_C_lsvm, max_iter=2000))
    multiclass_ovo_clf.fit(X_train, Y_train)
    Y_predict = multiclass_ovo_clf.predict(X_test)
    toc = time.time()
    ovo_time = toc - tic
    
    #Evaluation
    print('\n================================')
    print ("%-15s%-15s"%('Metric','Value'))
    print('--------------------------------')
    print("%-15s%-15f"%('Accuracy', metrics.accuracy_score(Y_test, Y_predict)))
    print("%-15s%-15f"%('Precision', metrics.precision_score(Y_test, Y_predict, average='weighted')))
    print("%-15s%-15f"%('Recall', metrics.recall_score(Y_test, Y_predict, average='weighted')))
    print('--------------------------------')
    print("Model: Linear SVM")
    print("Method: One Vs. One")
    print("Processed in %.6f seconds"%(ovo_time))
    print('================================\n')
    print("Classification Report:")
    print(metrics.classification_report(Y_test, Y_predict))
    
    print("Confusion Matrix")
    ovo_mat = metrics.confusion_matrix(Y_test, Y_predict)
    plt.figure(figsize=(12,12))
    plt.rc('font', size=20)
    sns.heatmap(ovo_mat, square=True, annot=True, fmt='d', cbar='False')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

#%% (4-2) Linear SVM OVR

svm_ovr_activate = False
if svm_ovr_activate:
    print("\n** Linear SVM One Vs. Rest **\n")
    tic = time.time()
    multiclass_ovr_clf = multiclass.OneVsRestClassifier(svm.LinearSVC(penalty='l2', C=best_C_lsvm, multi_class='ovr', max_iter=2000))
    multiclass_ovr_clf.fit(X_train, Y_train)
    Y_predict = multiclass_ovr_clf.predict(X_test)
    toc = time.time()
    ovr_time = toc - tic
    
    #Evaluation
    print('\n================================')
    print ("%-15s%-15s"%('Metric','Value'))
    print('--------------------------------')
    print("%-15s%-15f"%('Accuracy', metrics.accuracy_score(Y_test, Y_predict)))
    print("%-15s%-15f"%('Precision', metrics.precision_score(Y_test, Y_predict, average='weighted')))
    print("%-15s%-15f"%('Recall', metrics.recall_score(Y_test, Y_predict, average='weighted')))
    #print("%-15s%-15f"%('Log-Loss', metrics.log_loss(Y_test, Y_predict_prob, normalize=True)))
    print('--------------------------------')
    print("Model: Linear SVM")
    print("Method: One Vs. Rest")
    print("Processed in %.6f seconds"%(ovr_time))
    print('================================\n')
    print("Classification Report:")
    print(metrics.classification_report(Y_test, Y_predict))
    
    print("Confusion Matrix")
    ovr_mat = metrics.confusion_matrix(Y_test, Y_predict)
    plt.figure(figsize=(12,12))
    plt.rc('font', size=20)
    sns.heatmap(ovr_mat, square=True, annot=True, fmt='d', cbar='False')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


#%% (5-1) Kernel SVM - Polynomial

grid_poly_svm_activate = False
if grid_poly_svm_activate:
    print("\n** Grid Search for Polynomial SVM **\n")
    tic = time.time()
    print("Grid search ...")
    param_grid = {'C': 10**np.arange(-1.6,1,0.2,dtype=float), 'degree': [3,5,7,9]}
    grid = GridSearchCV(svm.SVC(kernel='poly'), param_grid, refit = True, verbose = 3)
    grid.fit(X_train, Y_train)
    grid_best_poly_svm = grid.best_params_
    toc = time.time()
    poly_svm_grid_time = toc - tic
    print("done!")
    print("processing time: %.6f sec"%poly_svm_grid_time)
    print("Best parameters found by grid search:")
    print(grid.best_params_)
    best_C_poly_svm = grid_best_poly_svm['C']
    best_degree_poly_svm = grid_best_poly_svm['degree']
    

#%% Polynomial SVM

poly_svm_activate = False
if poly_svm_activate:
    print("\n** SVM with Polynomial Kernel **")
    tic = time.time()
    poly_svm_clf = svm.SVC(kernel='poly', C=best_C_poly_svm, degree=best_degree_poly_svm, probability=True)
    poly_svm_clf.fit(X_train, Y_train)
    Y_predict = poly_svm_clf.predict(X_test)
    Y_predict_prob = poly_svm_clf.predict_proba(X_test)
    toc = time.time()
    poly_svm_time = toc - tic
    
    #Evaluation
    print('\n================================')
    print ("%-15s%-15s"%('Metric','Value'))
    print('--------------------------------')
    print("%-15s%-15f"%('Accuracy', metrics.accuracy_score(Y_test, Y_predict)))
    print("%-15s%-15f"%('Precision', metrics.precision_score(Y_test, Y_predict, average='weighted')))
    print("%-15s%-15f"%('Recall', metrics.recall_score(Y_test, Y_predict, average='weighted')))
    print("%-15s%-15f"%('Log-Loss', metrics.log_loss(Y_test, Y_predict_prob, normalize=True)))
    print('--------------------------------')
    print("Model: Support Vector Machine")
    print("Kernel: {}-Degree Polynomianl".format(best_degree_poly_svm))
    print("C value: {}".format(best_C_poly_svm))
    print("Processed in %.6f seconds"%(poly_svm_time))
    print('================================\n')
    print("Classification Report:")
    print(metrics.classification_report(Y_test, Y_predict))
    
    print("Confusion Matrix")
    poly_svm_mat = metrics.confusion_matrix(Y_test, Y_predict)
    plt.figure(figsize=(12,12))
    plt.rc('font', size=20)
    sns.heatmap(poly_svm_mat, square=True, annot=True, fmt='d', cbar='False')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    

#%% (5-2) Kernel SVM - RBF

grid_rbf_svm_activate = False
if grid_rbf_svm_activate:
    print("\n** Grid Search for RBF SVM **\n")
    tic = time.time()
    print("Grid search ...")
    param_grid = {'C': 10**np.arange(-1.5,0.9,0.4,dtype=float), 'gamma': 'scale'} 
    grid = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, refit = True, verbose = 3)
    grid.fit(X_train, Y_train)
    grid_best_rbf_svm = grid.best_params_
    toc = time.time()
    rbf_svm_grid_time = toc - tic
    print("done!")
    print("processing time: %.6f sec"%rbf_svm_grid_time)
    print("Best parameters found by grid search:")
    print(grid.best_params_)
    best_C_rbf_svm = grid_best_rbf_svm['C']


#%% RBF SVM

rbf_svm_activate = False
if rbf_svm_activate:
    print("\n** SVM with RBF Kernel **")
    tic = time.time()
    rbf_svm_clf = svm.SVC(kernel='rbf', C=best_C_rbf_svm, gamma='scale', probability=True)
    rbf_svm_clf.fit(X_train, Y_train)
    Y_predict = rbf_svm_clf.predict(X_test)
    Y_predict_prob = rbf_svm_clf.predict_proba(X_test)
    toc = time.time()
    rbf_svm_clf_time = toc - tic
    
    #Evaluation
    print('\n================================')
    print ("%-15s%-15s"%('Metric','Value'))
    print('--------------------------------')
    print("%-15s%-15f"%('Accuracy', metrics.accuracy_score(Y_test, Y_predict)))
    print("%-15s%-15f"%('Precision', metrics.precision_score(Y_test, Y_predict, average='weighted')))
    print("%-15s%-15f"%('Recall', metrics.recall_score(Y_test, Y_predict, average='weighted')))
    print("%-15s%-15f"%('Log-Loss', metrics.log_loss(Y_test, Y_predict_prob, normalize=True)))
    print('--------------------------------')
    print("Model: Support Vector Machine")
    print("Kernel: RBF (Gaussian)")
    print("C value: {}".format(best_C_rbf_svm))
    print("Processed in %.6f seconds"%(rbf_svm_clf_time))
    print('================================\n')
    print("Classification Report:")
    print(metrics.classification_report(Y_test, Y_predict))
    
    print("Confusion Matrix")
    rbf_svm_mat = metrics.confusion_matrix(Y_test, Y_predict)
    plt.figure(figsize=(12,12))
    plt.rc('font', size=20)
    sns.heatmap(rbf_svm_mat, square=True, annot=True, fmt='d', cbar='False')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    




