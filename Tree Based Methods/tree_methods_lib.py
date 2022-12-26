# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 01:19:29 2022

@author: mz52
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

#%% Loading Datasets

def load_dataset(print_info=True):
    features = ["Age", "Workclass", "Fnlwgt", "Education", "Education-Num", "MaritalStatus",
            "Occupation", "Relationship", "Race", "Sex", "CapitalGain", "CapitalLoss",
            "HoursPerWeek", "Country", "Income"]
    
    # download the data files
    train_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    test_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
    
    data_train = pd.read_csv(train_url, names=features, sep=r'\s*,\s*', engine='python', na_values="?")
    data_test = pd.read_csv(test_url, names=features, sep=r'\s*,\s*', engine='python', na_values="?", skiprows=1)
    if print_info:
        print('\n> Adult Dataset Info\n')
        data_train.info()
        print('\n> Statistical Summary\n')
        print(data_train.describe().T)
    return data_train, data_test, features


#%% Exploration

def data_exploration(dataset):
    plt.figure(figsize=(25,8))
    sns.countplot(y='Education', data = dataset, order = dataset.Education.value_counts().index)
    plt.title('Education')
    plt.xlabel('Number of Occurrences', fontsize=12)
    plt.ylabel('Degree', fontsize=12)
    
    plt.figure(figsize=(25,8))
    sns.countplot(y='Occupation',order= dataset.Occupation.value_counts().index, alpha=0.8,data = dataset)
    plt.title('Occupation')
    plt.xlabel('Number of Occurrences', fontsize=12)
    plt.ylabel('Job title', fontsize=12)
    
    plt.figure(figsize=(20,4))
    sns.countplot(y='Country',order=dataset.Country.value_counts().index[:10], alpha=0.8,data = dataset)
    plt.title('Countries')
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('Countries', fontsize=12)
    
    plt.figure(figsize=(20,4))
    plt.title('Race Distribution')
    plt.ylabel('Race', fontsize=12)
    sns.countplot(y = 'Race', alpha=0.8, data=dataset)
    
    plt.figure(figsize=(10,6))
    sns.barplot(x = 'Sex',y='Age' ,hue='Income', alpha=0.8, data=dataset)
    
    plt.figure(figsize=(30,10))
    sns.countplot(x = 'Occupation',hue='Income', alpha=0.8, data=dataset)
    
    plt.figure(figsize=(30,10))
    sns.countplot(x = 'MaritalStatus',hue='Income', alpha=0.8, data=dataset)
    
    plt.figure(figsize=(30,10))
    plt.title('Age vs Income')
    plt.xlabel('Age', fontsize=15)
    sns.countplot(x = 'Age',hue='Income', data=dataset)
    
    plt.figure(figsize=(20,4))
    density = sns.FacetGrid(dataset, col = "Income",height=5)
    density.map(sns.kdeplot, "Age")
    
    # correlation Heatmap
    fig, (ax) = plt.subplots(1, 1, figsize=(10,6))
    hm = sns.heatmap(dataset.corr(), ax=ax, cmap="coolwarm", square=True, annot=True, fmt='.2f', annot_kws={"size": 14}, linewidths=.05)
    fig.subplots_adjust(top=0.93)
    fig.suptitle('US DATA INCOME Correlation Heatmap', fontsize=14, fontweight='bold')
    
    plt.show()

#%% Preparing Datasets

def preparing_dataset(original_train, original_test):
    labels_train = original_train['Income']
    labels_train = labels_train.replace('<=50K', 0).replace('>50K', 1)
    labels_train = labels_train.replace('<=50K.', 0).replace('>50K.', 1)
    Y_train = labels_train
    labels_test = original_test['Income']
    labels_test = labels_test.replace('<=50K', 0).replace('>50K', 1)
    labels_test = labels_test.replace('<=50K.', 0).replace('>50K.', 1)
    Y_test = labels_test
    
    # Redundant column
    del original_train["Education"]
    del original_test["Education"]
    # Remove income variable
    del original_train["Income"]
    del original_test["Income"]
    
    binary_data_train = pd.get_dummies(original_train)
    del binary_data_train['Country_Holand-Netherlands']
    binary_data_test = pd.get_dummies(original_test)
    
    #feature_cols_train = binary_data_train[binary_data_train.columns[:-2]]
    feature_cols_train = binary_data_train[binary_data_train.columns]
    #feature_cols_test = binary_data_test[binary_data_test.columns[:-2]]
    feature_cols_test = binary_data_test[binary_data_test.columns]
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(feature_cols_train), columns=feature_cols_train.columns)
    X_test = pd.DataFrame(scaler.transform(feature_cols_test), columns=feature_cols_test.columns)
    return X_train, X_test, Y_train, Y_test















