
"""
Created on Wed Aug 31 09:39:43 2022

@author: mz52
"""

from sklearn.datasets import make_moons, make_circles, make_classification
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
import numpy as np

cm_bright = ListedColormap(['#FF0000', '#0000FF'])

def generate_datasets(n_samples):
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1, n_samples=n_samples)
    linearly_separable = (X, y)
    datasets = [linearly_separable, make_moons(noise=0.15, random_state=0, n_samples=n_samples), make_moons(noise=0.35, random_state=0, n_samples=n_samples), make_circles(noise=0.25, factor=0.5, random_state=1, n_samples=n_samples)]
    return datasets    
    

def plot_datasets(datasets):
    # Visualize the datasets
    plt.rc('font',size=20)
    fig_x = int(len(datasets)*10)
    plt.figure(figsize=(fig_x,8))
    for ds_cnt, ds in enumerate(datasets):
        X, y = ds
        X = StandardScaler().fit_transform(X)
        ax = plt.subplot(1, len(datasets), ds_cnt+1)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, s=150, edgecolors='k', alpha=0.65)
    
        

def separate_train_test(datasets, split_coef):
    n_samps = datasets[0][0].shape[0]
    training_portion = int(split_coef/100 * n_samps)
    datasets_train = []
    datasets_test = []
    for i in range(len(datasets)):
        train_i = (datasets[i][0][:training_portion,:], datasets[i][1][:training_portion])
        test_i = (datasets[i][0][training_portion:,:], datasets[i][1][training_portion:])
        datasets_train.append(train_i)
        datasets_test.append(test_i)
    return datasets_train, datasets_test



def my_knn(K, X_train, Y_train, X_test):
    Y_test = np.zeros(len(Y_train)) - 1
    for test_idx in range(len(X_test)):
        x = X_test[test_idx,:]
        dist = np.sqrt(np.sum((X_train-x)**2, axis=1))
        y_neighbors = 0
        dist_sweep = dist
        for n in range(K):
            neighbors = np.argmin(dist_sweep)
            y_neighbors = y_neighbors + Y_train[neighbors]
            dist_sweep = np.delete(dist_sweep, neighbors)
        threshold = y_neighbors / K
        if (threshold > 0.5):
            Y_test[test_idx] = 1
        else:
            Y_test[test_idx] = 0
    return Y_test


def knn_error(datasets_train, datasets_test):
    print("Measuring test and training error for KNN")
    n_datasets = len(datasets_train)
    K_vec = np.arange(0,210,10)
    K_vec[0] = 1
    train_err = np.zeros((len(K_vec),n_datasets))
    test_err = np.zeros((len(K_vec),n_datasets))
    for k_sw in range(len(K_vec)):
        K = K_vec[k_sw]
        print("Processind ... K = {}".format(K))
        for dataset_idx in range(n_datasets):
            X_train = datasets_train[dataset_idx][0]
            Y_train = datasets_train[dataset_idx][1]
        
            X_test = datasets_test[dataset_idx][0]
            Y_test = datasets_test[dataset_idx][1]

            Y_out_train = my_knn(K, X_train, Y_train, X_train)
            Y_out_test = my_knn(K, X_train, Y_train, X_test)
            
            train_err[k_sw,dataset_idx] = np.mean(np.abs(Y_out_train - Y_train))
            test_err[k_sw,dataset_idx] = np.mean(np.abs(Y_out_test - Y_test))
    print("Error Measured!")
    for i in range(n_datasets):
        plt.figure(figsize=(10,8))
        plt.plot(K_vec, train_err[:,i], lw=2, label='Training Error')
        plt.plot(K_vec, test_err[:,i], lw=2, label='Test Error')
        plt.legend(loc='lower left', fontsize=20)
        plt.xlabel("Model Complexity in terms of K")
        plt.ylabel("Model Error")
        plt.title("Dataset #{}".format(i))
        plt.gca().invert_xaxis()
        plt.show()
    print("Finish plotting!")


def knn_showcase(K, datasets_train, datasets_test):
    n_datasets = len(datasets_train)
    knn_out = []
    for dataset_idx in range(n_datasets):
        X_train = datasets_train[dataset_idx][0]
        Y_train = datasets_train[dataset_idx][1]
        train_set = [(X_train, Y_train)]
        X_test = datasets_test[dataset_idx][0]
        Y_test = datasets_test[dataset_idx][1]
        test_set = [(X_test, Y_test)]
        K = 5
        Y_out = my_knn(K, X_train, Y_train, X_train)
        knn_out.append((X_train, Y_out))
    
    plot_datasets(knn_out)
    plt.suptitle("Training Sets with K = {}".format(K), fontsize=36)

def main():
    n_samples = 1000
    datasets = generate_datasets(n_samples)
    n_datasets = len(datasets)
    datasets_train, datasets_test = separate_train_test(datasets, split_coef=50) # split_coef in [0,100] indicates the training portion of dataset in percentage
    
    K = 5
    #knn_showcase(K, datasets_train, datasets_test)
    
    knn_error(datasets_train, datasets_test)
    
            

    
if __name__ == '__main__':
    main()
