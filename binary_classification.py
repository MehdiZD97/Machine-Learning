# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 11:50:17 2022

@author: mz52
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests, gzip, os, hashlib
from sklearn import svm, metrics
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
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

X_train_total_mat = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28,28))
Y_train_total = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
X_test_total_mat = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28,28))
Y_test_total = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]

# scaling
X_train_total = (X_train_total_mat/255) if scale_data else X_train_total_mat
X_test_total = (X_test_total_mat/255) if scale_data else X_test_total_mat
# flattenning
X_train_total = X_train_total.reshape(len(X_train_total),28*28) if flatten_data else X_train_total
X_test_total = X_test_total.reshape(len(X_test_total),28*28) if flatten_data else X_test_total

scaler = StandardScaler()
X_train_total = scaler.fit_transform(X_train_total)
X_test_total = scaler.transform(X_test_total)

#%% Binary Classifier for digits 3 & 8

idx_train_bclf = np.sort(np.append(np.where((Y_train_total==3))[0],np.where((Y_train_total==8))[0]))
idx_test_bclf = np.sort(np.append(np.where((Y_test_total==3))[0],np.where((Y_test_total==8))[0]))
X_train = X_train_total[idx_train_bclf]
Y_train = Y_train_total[idx_train_bclf]
X_test = X_test_total[idx_test_bclf]
Y_test = Y_test_total[idx_test_bclf]

print("** Preprocessing **")
print("%-25s%-25s"%('Dataset:','MNIST Handwritten Digits Dataset'))
print("%-24s"%('Dataset dimensions:'), X_train.shape)
print("%-25s%-25s"%('Classes:','Digit 3 & Digit 8 (Binary Classification)'))


#%% Linear SVM

linear_svm_activate = False
if linear_svm_activate:
    print("\n** Binary Classifier with Linear SVM **")
    tic = time.time()
    C = 0.001
    svm_bin_clf = svm.SVC(kernel='linear', C=C, probability=True)
    svm_bin_clf.fit(X_train, Y_train)
    Y_predict = svm_bin_clf.predict(X_test)
    Y_predict_prob = svm_bin_clf.predict_proba(X_test)
    toc = time.time()
    lsvm_time = toc - tic
    
    # Evaluation
    print('\n================================')
    print ("%-15s%-15s"%('Metric','Value'))
    print('--------------------------------')
    print("%-15s%-15f"%('Accuracy', metrics.accuracy_score(Y_test, Y_predict)))
    print("%-15s%-15f"%('Precision', metrics.precision_score(Y_test, Y_predict, pos_label=3)))
    print("%-15s%-15f"%('Recall', metrics.recall_score(Y_test, Y_predict, pos_label=3)))
    print("%-15s%-15f"%('Log-Loss', metrics.log_loss(Y_test, Y_predict_prob, normalize=True)))
    print('--------------------------------')
    print("Model: Linear SVM")
    print("C value: {}".format(C))
    print("Processed in %.6f seconds"%(lsvm_time))
    print('================================')

#%% K-Fold Cross-Validation for SVM - based on score

svm_cv_activate = False
if svm_cv_activate:
    K = 5
    print("\n** Linear SVM with {}-Fold Cross-Validation **\n".format(K))
    tic = time.time()
    np.random.seed(0)
    rand=np.arange(X_train.shape[0])
    np.random.shuffle(rand)
    cv_portion = int(X_train.shape[0]/K)
    C_vec_svm = 10**np.arange(-2.5,0.52,0.04,dtype=float)
    #C_vec_svm = 10**np.arange(-2,2,1,dtype=float)
    score_mat_val = np.zeros((K,len(C_vec_svm)))
    score_mat_train = np.zeros((K,len(C_vec_svm)))
    score_mat_test = np.zeros((K,len(C_vec_svm)))
    for fold in range(K):
        print("-- processing for fold = {}".format(fold))
        idx_val_cv = rand[fold*cv_portion:(fold+1)*cv_portion]
        idx_train_cv = np.delete(rand, np.arange(fold*cv_portion,(fold+1)*cv_portion))
        X_train_cv = X_train[idx_train_cv]
        Y_train_cv = Y_train[idx_train_cv]
        X_val_cv = X_train[idx_val_cv]
        Y_val_cv = Y_train[idx_val_cv]
        for it in range(len(C_vec_svm)):
            print("  -> C value idx: {} / {}".format(it+1,len(C_vec_svm)))
            svm_bin_clf = svm.SVC(kernel='linear', C=C_vec_svm[it])
            svm_bin_clf.fit(X_train_cv, Y_train_cv)
            score_mat_val[fold,it] = svm_bin_clf.score(X_val_cv,Y_val_cv)
            score_mat_train[fold,it] = svm_bin_clf.score(X_train_cv,Y_train_cv)
            score_mat_test[fold,it] = svm_bin_clf.score(X_test,Y_test)
    best_C_svm = C_vec_svm[np.argmax(np.mean(score_mat_val,axis=0))]
    toc = time.time()
    cv_time = toc - tic
    print("done processing!")
    print("processing time: %.6f sec"%cv_time)
    print("Best Value for Hyperparameter C: {}".format(best_C_svm))
    filename_val = "SVM_CV_" + str(K) + "_fold_val_score_mat.npy"
    filename_train = "SVM_CV_" + str(K) + "_fold_train_score_mat.npy"
    filename_test = "SVM_CV_" + str(K) + "_fold_test_score_mat.npy"
    np.save(filename_val,score_mat_val)
    np.save(filename_train,score_mat_train)
    np.save(filename_test,score_mat_test)


#%% Evaluating SVM with 5-Fold CV

evaluate_svm_cv_activate = True
if evaluate_svm_cv_activate:
    path = 'C:/Users/mz52/MATLAB-Drive/2-1 Fall 2022/ELEC 578/'
    #path = ''
    score_mat_val = np.load(path + 'SVM_CV_5_fold_val_score_mat.npy')
    score_mat_train = np.load(path + 'SVM_CV_5_fold_train_score_mat.npy')
    score_mat_test = np.load(path + 'SVM_CV_5_fold_test_score_mat.npy')
    
    plt.figure(figsize=(10,6))
    plt_min = 0
    plt_max = 76
    plt.plot(C_vec_svm[plt_min:plt_max], 1-np.mean(score_mat_train,axis=0)[plt_min:plt_max], '-', lw=2.5, label='Train')
    plt.plot(C_vec_svm[plt_min:plt_max], 1-np.mean(score_mat_test,axis=0)[plt_min:plt_max], '--', lw=2.5, label='Test')
    plt.plot(C_vec_svm[plt_min:plt_max], 1-np.mean(score_mat_val,axis=0)[plt_min:plt_max], '-.', lw=2.5, label='Validation')
    plt.stem(C_vec_svm[np.argmax(np.mean(score_mat_val,axis=0))],1-np.mean(score_mat_val,axis=0)[np.argmax(np.mean(score_mat_val,axis=0))], linefmt='r--', markerfmt='ro', label='Minimum Error')
    plt.legend(loc='best',fontsize=16)
    plt.xlabel("C Values")
    plt.ylabel("Error")


#%% SVM with Built-in CV Function

svm_cv_grid_activate = True
if svm_cv_grid_activate:
    print("\n** Grid Search for Linear SVM Parameter Tuning **\n")
    param_grid = {'C': 10**np.arange(-2,1.4,0.2,dtype=float), 'kernel': ['linear']} 
    grid = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 3)
    grid.fit(X_train, Y_train)
    grid_best_svm = grid.best_params_
    print("Best parameters found by grid search:")
    print(grid.best_params_)


#%% SVM Model with the Best Hyperparameter

svm_best_C_activate = True
if svm_best_C_activate:
    print("\n** Binary Classifier with Best Linear SVM **")
    tic = time.time()
    best_C_svm = C_vec_svm[np.argmax(np.mean(score_mat_val,axis=0))]
    svm_bin_clf = svm.SVC(kernel='linear', C=best_C_svm, probability=True)
    svm_bin_clf.fit(X_train, Y_train)
    Y_predict = svm_bin_clf.predict(X_test)
    Y_predict_prob = svm_bin_clf.predict_proba(X_test)
    toc = time.time()
    best_lsvm_time = toc - tic
    
    # Evaluation
    print('\n================================')
    print ("%-15s%-15s"%('Metric','Value'))
    print('--------------------------------')
    print("%-15s%-15f"%('Accuracy', metrics.accuracy_score(Y_test, Y_predict)))
    print("%-15s%-15f"%('Precision', metrics.precision_score(Y_test, Y_predict, pos_label=3)))
    print("%-15s%-15f"%('Recall', metrics.recall_score(Y_test, Y_predict, pos_label=3)))
    print("%-15s%-15f"%('Log-Loss', metrics.log_loss(Y_test, Y_predict_prob, normalize=True)))
    print('--------------------------------')
    print("Model: Linear SVM with Best C")
    print("C value: {}".format(best_C_svm))
    print("Processed in %.6f seconds"%(best_lsvm_time))
    print('================================\n')
    print("Classification Report:")
    print(metrics.classification_report(Y_test, Y_predict))
    
    print("Confusion Matrix")
    svm_mat = metrics.confusion_matrix(Y_test, Y_predict)
    plt.figure(figsize=(10,10))
    plt.rc('font', size=20)
    sns.heatmap(svm_mat, square=True, annot=True, fmt='d', cbar='False', xticklabels=['Digit 3','Digit 8'], yticklabels=['Digit 3','Digit 8'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    print("Pixel Heatmap")
    plt.figure(figsize=(10,10))
    plt.rc('font', size=14)
    svm_coef = svm_bin_clf.coef_
    svm_coef = np.abs(np.reshape(svm_coef.flatten(),(28,28)))
    sns.heatmap(svm_coef,square=True)
    plt.xlabel('Pixel Index')
    plt.ylabel('Pixel Index')
    plt.show()


#%% Logistic Regression

logistic_regression_activate = False
if logistic_regression_activate:
    print("\n** Binary Classifier with Logistic Regression **")
    tic = time.time()
    C = 0.001
    lr_bin_clf = linear_model.LogisticRegression(penalty='l2',C=C)
    lr_bin_clf.fit(X_train, Y_train)
    Y_predict = lr_bin_clf.predict(X_test)
    Y_predict_prob = lr_bin_clf.predict_proba(X_test)
    toc = time.time()
    lr_time = toc - tic
    
    # Evaluation
    print('\n================================')
    print ("%-15s%-15s"%('Metric','Value'))
    print('--------------------------------')
    print("%-15s%-15f"%('Accuracy', metrics.accuracy_score(Y_test, Y_predict)))
    print("%-15s%-15f"%('Precision', metrics.precision_score(Y_test, Y_predict, pos_label=3)))
    print("%-15s%-15f"%('Recall', metrics.recall_score(Y_test, Y_predict, pos_label=3)))
    print("%-15s%-15f"%('Log-Loss', metrics.log_loss(Y_test, Y_predict_prob, normalize=True)))
    print('--------------------------------')
    print("Model: Logistic Regression")
    print("C value: {}".format(C))
    print("Processed in %.6f seconds"%(lr_time))
    print('================================')


#%% K-Fold Cross-Validation for Logistic Regression

lr_cv_activate = False
if lr_cv_activate:
    K = 5
    print("\n** Logistic Regression with {}-Fold Cross-Validation **\n".format(K))
    tic = time.time()
    np.random.seed(0)
    rand=np.arange(X_train.shape[0])
    np.random.shuffle(rand)
    cv_portion = int(X_train.shape[0]/K)
    C_vec_lr = 10**np.arange(-2,1.02,0.02,dtype=float)
    log_loss_mat_val = np.zeros((K,len(C_vec_lr)))
    log_loss_mat_train = np.zeros((K,len(C_vec_lr)))
    log_loss_mat_test = np.zeros((K,len(C_vec_lr)))
    for fold in range(K):
        print("-- processing for fold = {}".format(fold))
        idx_val_cv = rand[fold*cv_portion:(fold+1)*cv_portion]
        idx_train_cv = np.delete(rand, np.arange(fold*cv_portion,(fold+1)*cv_portion))
        X_train_cv = X_train[idx_train_cv]
        Y_train_cv = Y_train[idx_train_cv]
        X_val_cv = X_train[idx_val_cv]
        Y_val_cv = Y_train[idx_val_cv]
        for it in range(len(C_vec_lr)):
            lr_bin_clf = linear_model.LogisticRegression(penalty='l2',C=C_vec_lr[it],max_iter=500)
            lr_bin_clf.fit(X_train_cv, Y_train_cv)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                log_loss_mat_val[fold,it] = metrics.log_loss(Y_val_cv, lr_bin_clf.predict_proba(X_val_cv))
                log_loss_mat_train[fold,it] = metrics.log_loss(Y_train_cv, lr_bin_clf.predict_proba(X_train_cv))
                log_loss_mat_test[fold,it] = metrics.log_loss(Y_test, lr_bin_clf.predict_proba(X_test))
    best_C_lr = C_vec_lr[np.argmin(np.mean(log_loss_mat_val,axis=0))]
    toc = time.time()
    lr_cv_time = toc - tic
    print("done processing!")
    print("processing time: %.6f sec"%lr_cv_time)
    print("Best Value for Hyperparameter C: {}".format(best_C_lr))
    filename_val = "LR_CV_" + str(K) + "_fold_val_log_loss_mat.npy"
    filename_train = "LR_CV_" + str(K) + "_fold_train_log_loss_mat.npy"
    filename_test = "LR_CV_" + str(K) + "_fold_test_log_loss_mat.npy"
    np.save(filename_val,log_loss_mat_val)
    np.save(filename_train,log_loss_mat_train)
    np.save(filename_test,log_loss_mat_test)


#%% Evaluating Logistic Regression with 5-Fold CV

evaluate_lr_cv_activate = True
if evaluate_lr_cv_activate:
    path = 'C:/Users/mz52/MATLAB-Drive/2-1 Fall 2022/ELEC 578/'
    #path = ''
    log_loss_mat_val = np.load(path + 'LR_CV_5_fold_val_log_loss_mat.npy')
    log_loss_mat_train = np.load(path + 'LR_CV_5_fold_train_log_loss_mat.npy')
    log_loss_mat_test = np.load(path + 'LR_CV_5_fold_test_log_loss_mat.npy')
    
    plt.figure(figsize=(10,6))
    plt_min = 1
    plt_max = 120
    plt.plot(C_vec_lr[plt_min:plt_max], np.mean(log_loss_mat_train,axis=0)[plt_min:plt_max], '-', lw=2.5, label='Train')
    plt.plot(C_vec_lr[plt_min:plt_max], np.mean(log_loss_mat_test,axis=0)[plt_min:plt_max], '--', lw=2.5, label='Test')
    plt.plot(C_vec_lr[plt_min:plt_max], np.mean(log_loss_mat_val,axis=0)[plt_min:plt_max], '-.', lw=2.5, label='Validation')
    plt.stem(C_vec_lr[np.argmin(np.mean(log_loss_mat_val,axis=0))],np.mean(log_loss_mat_val,axis=0)[np.argmin(np.mean(log_loss_mat_val,axis=0))], linefmt='r--', markerfmt='ro', label='Minimum Error')
    plt.legend(loc='best',fontsize=16)
    plt.xlabel("C Values")
    plt.ylabel("Error")


#%% LR with Built-in CV Function

lr_cv_grid_activate = True
if lr_cv_grid_activate:
    print("\n** Grid Search for Logistic Regression Parameter Tuning **\n")
    param_grid = {'C': 10**np.arange(-2,1.2,0.2,dtype=float)} 
    grid = GridSearchCV(linear_model.LogisticRegression(penalty='l2',max_iter=500), param_grid, refit = True, verbose = 3)
    grid.fit(X_train, Y_train)
    grid_best_lr = grid.best_params_
    print("Best parameters found by grid search:")
    print(grid.best_params_)
    

#%% LR Model with the Best Hyperparameter

lr_best_C_activate = True
if lr_best_C_activate:
    print("\n** Binary Classifier with Best Logistic Regression **")
    tic = time.time()
    #best_C_lr = C_vec_lr[np.argmin(np.mean(log_loss_mat_val,axis=0))]
    best_C_lr = grid_best_lr['C']
    lr_bin_clf = linear_model.LogisticRegression(penalty='l2',C=best_C_lr,max_iter=500)
    lr_bin_clf.fit(X_train, Y_train)
    Y_predict = lr_bin_clf.predict(X_test)
    Y_predict_prob = lr_bin_clf.predict_proba(X_test)
    toc = time.time()
    best_lr_time = toc - tic
    
    # Evaluation
    print('\n================================')
    print ("%-15s%-15s"%('Metric','Value'))
    print('--------------------------------')
    print("%-15s%-15f"%('Accuracy', metrics.accuracy_score(Y_test, Y_predict)))
    print("%-15s%-15f"%('Precision', metrics.precision_score(Y_test, Y_predict, pos_label=3)))
    print("%-15s%-15f"%('Recall', metrics.recall_score(Y_test, Y_predict, pos_label=3)))
    print("%-15s%-15f"%('Log-Loss', metrics.log_loss(Y_test, Y_predict_prob, normalize=True)))
    print('--------------------------------')
    print("Model: Logistic Regression with Best C")
    print("C value: {}".format(best_C_lr))
    print("Processed in %.6f seconds"%(best_lr_time))
    print('================================\n')
    print("Classification Report:")
    print(metrics.classification_report(Y_test, Y_predict))
    
    print("Confusion Matrix")
    lr_mat = metrics.confusion_matrix(Y_test, Y_predict)
    plt.figure(figsize=(10,10))
    plt.rc('font', size=20)
    sns.heatmap(lr_mat, square=True, annot=True, fmt='d', cbar='False', xticklabels=['Digit 3','Digit 8'], yticklabels=['Digit 3','Digit 8'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    print("Pixel Heatmap")
    plt.figure(figsize=(10,10))
    plt.rc('font', size=14)
    lr_coef = lr_bin_clf.coef_
    lr_coef = np.abs(np.reshape(lr_coef.flatten(),(28,28)))
    sns.heatmap(lr_coef,square=True)
    plt.xlabel('Pixel Index')
    plt.ylabel('Pixel Index')
    plt.show()
    





