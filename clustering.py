# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 19:30:58 2022

@author: mz52
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.cluster import AgglomerativeClustering, FeatureAgglomeration
from sklearn.cluster import SpectralBiclustering, SpectralCoclustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn import metrics
import seaborn as sns

def encircle(x,y, ax=None, **kw):
    if not ax: ax=plt.gca()
    p = np.c_[x,y]
    hull = ConvexHull(p)
    poly = plt.Polygon(p[hull.vertices,:], **kw)
    ax.add_patch(poly)

#%% Loading the dataset

run_on_server = False
if run_on_server:
    path = ""
else:
    path = 'C:/Users/mz52/MATLAB-Drive/2-1 Fall 2022/ELEC 578/HW3/'
dataset = pd.read_csv(path+"authors.csv")

feature_names = dataset.columns
X = dataset.iloc[:,1:]
auth = dataset.iloc[:,0]
auth = auth.rename('Name')
auth_label = [0 if elem=='Austen' else (1 if elem=='London' else (2 if elem=='Milton' else 3)) for elem in auth]
auth_label = np.array(auth_label)
    
# scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("** Preprocessing **")
print("%-25s%-25s"%('Dataset:','Book Word Count Dataset'))
print("%-24s"%('Dataset dimensions:'), X.shape)
print("%-25s%-25s"%('Authors:','Austen, London, Milton, Shakespeare'))

#%% Dimension Reduction for Visualization

pca_obj = PCA(n_components=2)
X_pca = pca_obj.fit_transform(X_scaled)

#%% K-Means

K_km = 4
print('\nApplying K-Means Clustering')
print('--------------------------------------')
print("%-28s%-15i"%('# of Clusters(K):',K_km))
km_obj = KMeans(n_clusters=K_km, init='k-means++', n_init=100, max_iter=500, random_state=5)
km_labels = km_obj.fit_predict(X_scaled)
km_ssd = km_obj.inertia_
print("%-28s%-15.4f"%('Sum of Squared Distances:',km_ssd))

auth_label_km = np.asarray([0 if elem=='Austen' else (1 if elem=='London' else (2 if elem=='Milton' else 3)) for elem in auth])
#km_precision = ((len(km_labels) - (km_labels != auth_label_km).sum()) / len(km_labels) * 100)
km_precision = metrics.rand_score(km_labels, auth_label_km) * 100
km_precision_mi = metrics.adjusted_mutual_info_score(km_labels, auth_label_km) # based on mutual info
print("%-28s%.4f%s"%('Precision:',km_precision,' %'))

#%%

plt.figure(figsize=(16,6))
plt.rc('font', size=14)
plt.subplot(1,2,1)
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=km_labels)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-Means with K = {}'.format(K_km))
plt.subplot(1,2,2)
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=auth_label_km)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('True Clusters')

#%%

k_max = 15
km_ssd_vec = np.zeros(k_max)
for k in range(1,k_max+1):
    km_obj = KMeans(n_clusters=k, init='k-means++', n_init=100, max_iter=500)
    km_obj.fit(X_scaled)
    km_ssd_vec[k-1] = km_obj.inertia_

#%%

plt.figure(figsize=(10,6))
plt.rc('font', size=14)
plt.plot(np.arange(1,k_max+1), km_ssd_vec, '-o', lw=3)
plt.grid(axis='y',alpha=0.5)
plt.xticks(np.arange(1,k_max+1))
plt.ylim([35000,60000])
plt.vlines(3,35000,km_ssd_vec[2],linestyle='--',color='r')
plt.vlines(4,35000,km_ssd_vec[3],linestyle='--',color='r')
plt.xlabel('K (# of clusters)')
plt.ylabel('Sum of Squared Distances')

#%% Hierarchical (Agglomerative) Clustering

K_hc = 4
dist = 'euclidean' # dist: 'euclidean' / 'l1' / 'l2' / 'manhattan' / 'cosine'
linkage = 'ward' # linkage: 'ward' / 'complete' / 'average' / 'single'
print('\nApplying Hierarchical Clustering')
print('--------------------------------')
print("%-20s%-15i"%('# of Clusters(K):',K_hc))
print("%-20s%-15s"%('Distance Metric:',dist))
print("%-20s%-15s"%('Linkage:',linkage))
hc_obj = AgglomerativeClustering(n_clusters=K_hc, affinity=dist,compute_distances=True, linkage=linkage)
hc_labels = hc_obj.fit_predict(X_scaled)
auth_label_hc = np.asarray([2 if elem=='Austen' else (0 if elem=='London' else (3 if elem=='Milton' else 1)) for elem in auth])
#hc_precision = ((len(hc_labels) - (hc_labels != auth_label_hc).sum()) / len(hc_labels) * 100)
hc_precision = metrics.rand_score(hc_labels, auth_label_hc) * 100
hc_precision_mi = metrics.adjusted_mutual_info_score(hc_labels, auth_label_hc) # based on mutual info
print("%-20s%.4f%s"%('Precision:',hc_precision,' %'))

#%%
    
plt.figure(figsize=(16,6))
plt.rc('font', size=14)
plt.subplot(1,2,1)
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=hc_labels)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Agglomerative with K = {} / {} / {}'.format(K_hc,dist,linkage))
cl0 = X_pca[hc_labels==0,:]
cl1 = X_pca[hc_labels==1,:]
cl2 = X_pca[hc_labels==2,:]
cl3 = X_pca[hc_labels==3,:]
alpha = 0.1
encircle(cl0[:,0], cl0[:,1], ec="k", fc="r", alpha=alpha)
encircle(cl1[:,0], cl1[:,1], ec="k", fc="b", alpha=alpha)
encircle(cl2[:,0], cl2[:,1], ec="k", fc="g", alpha=alpha)
encircle(cl3[:,0], cl3[:,1], ec="k", fc="y", alpha=alpha)

plt.subplot(1,2,2)
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=auth_label_hc)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('True Clusters')
cl0 = X_pca[auth_label_hc==0,:]
cl1 = X_pca[auth_label_hc==1,:]
cl2 = X_pca[auth_label_hc==2,:]
cl3 = X_pca[auth_label_hc==3,:]
alpha = 0.1
encircle(cl0[:,0], cl0[:,1], ec="k", fc="r", alpha=alpha)
encircle(cl1[:,0], cl1[:,1], ec="k", fc="b", alpha=alpha)
encircle(cl2[:,0], cl2[:,1], ec="k", fc="g", alpha=alpha)
encircle(cl3[:,0], cl3[:,1], ec="k", fc="y", alpha=alpha)

#%%
# Plotting Dendrogram
from scipy.cluster.hierarchy import dendrogram
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

plt.figure(figsize=(14,8))
plt.rc('font', size=14)
plot_dendrogram(hc_obj, truncate_mode="level", p=5)
plt.grid(axis='y',alpha=0.3)
plt.xlabel('Observation Index')
plt.ylabel('Height Level')

#%% Spectral Clustering

K_sc = 4
affinity = 'nearest_neighbors' # affinity: 'rbf' / 'nearest_neighbors'
print('\nApplying Spectral Clustering')
print('----------------------------')
print("%-20s%-15i"%('# of Clusters(K):',K_sc))
print("%-20s%-15s"%('Affinity:',affinity))
sc_obj = SpectralClustering(n_clusters=K_sc, n_init=50, affinity=affinity, gamma=1, random_state=5)
sc_labels = sc_obj.fit_predict(X_scaled)
auth_label_sc = np.asarray([1 if elem=='Austen' else (2 if elem=='London' else (3 if elem=='Milton' else 0)) for elem in auth])
#sc_precision = ((len(sc_labels) - (sc_labels != auth_label_sc).sum()) / len(sc_labels) * 100)
sc_precision = metrics.rand_score(sc_labels, auth_label_sc) * 100
sc_precision_mi = metrics.adjusted_mutual_info_score(sc_labels, auth_label_sc) # based on mutual info
print("%-20s%.4f%s"%('Precision:',sc_precision,' %'))

#%%

plt.figure(figsize=(16,6))
plt.rc('font', size=14)
plt.subplot(1,2,1)
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=sc_labels)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Spectral Clustering with K = {}'.format(K_sc))
cl0 = X_pca[sc_labels==0,:]
cl1 = X_pca[sc_labels==1,:]
cl2 = X_pca[sc_labels==2,:]
cl3 = X_pca[sc_labels==3,:]
alpha = 0.1
encircle(cl0[:,0], cl0[:,1], ec="k", fc="r", alpha=alpha)
encircle(cl1[:,0], cl1[:,1], ec="k", fc="b", alpha=alpha)
encircle(cl2[:,0], cl2[:,1], ec="k", fc="g", alpha=alpha)
encircle(cl3[:,0], cl3[:,1], ec="k", fc="y", alpha=alpha)

plt.subplot(1,2,2)
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=auth_label_sc)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('True Clusters')
cl0 = X_pca[auth_label_sc==0,:]
cl1 = X_pca[auth_label_sc==1,:]
cl2 = X_pca[auth_label_sc==2,:]
cl3 = X_pca[auth_label_sc==3,:]
alpha = 0.1
encircle(cl0[:,0], cl0[:,1], ec="k", fc="r", alpha=alpha)
encircle(cl1[:,0], cl1[:,1], ec="k", fc="b", alpha=alpha)
encircle(cl2[:,0], cl2[:,1], ec="k", fc="g", alpha=alpha)
encircle(cl3[:,0], cl3[:,1], ec="k", fc="y", alpha=alpha)

#%%

K_fa = 4
dist = 'euclidean' # dist: 'euclidean' / 'l1' / 'l2' / 'manhattan' / 'cosine'
linkage = 'ward' # linkage: 'ward' / 'complete' / 'average' / 'single'
print('\nApplying Feature Agglomeration')
print('--------------------------------')
print("%-20s%-15i"%('# of Clusters(K):',K_fa))
print("%-20s%-15s"%('Distance Metric:',dist))
print("%-20s%-15s"%('Linkage:',linkage))
fa_obj = FeatureAgglomeration(n_clusters=K_fa, affinity=dist, compute_distances=True, linkage=linkage)
fa_obj.fit(X_scaled)
fa_labels = fa_obj.labels_

plt.figure(figsize=(20,8))
plt.rc('font', size=14)
a = plot_dendrogram(fa_obj, truncate_mode="level", p=10)
plt.grid(axis='y',alpha=0.3)
plt.xlabel('Feature Index')
plt.ylabel('Height Level')

#%% Spectral Bi-Clustering

K_spbi = 4
print("\nApplying Spectral Bi-Clustering")
X_clust = X_scaled
spbi_obj = SpectralBiclustering(n_clusters=K_spbi, method='log', init='k-means++', random_state=5) # method = 'bistochastic'/'scale'/'log'
spbi_obj.fit(X_clust)
print("Number of clusters: K = {}".format(K_spbi))

X_spbi = X_clust[np.argsort(spbi_obj.row_labels_)]
X_spbi = X_spbi[:, np.argsort(spbi_obj.column_labels_)]
features_spbi = feature_names.values[1:][np.argsort(spbi_obj.column_labels_)]
df_spbi = pd.DataFrame(data=X_spbi, columns=features_spbi)
plt.figure(figsize=(22,15))
sns.heatmap(df_spbi, cmap=plt.cm.Blues)
plt.xlabel('Features', fontsize=20)
plt.ylabel('Observations', fontsize=20)

#%% Spectral Co-Clustering

K_spco = 4
print("\nApplying Spectral Co-Clustering")
X_clust = X_scaled
spco_obj = SpectralCoclustering(n_clusters=4, random_state=5)
spco_obj.fit(X_clust)
print("Number of clusters: K = {}".format(K_spco))

X_spco = X_clust[np.argsort(spco_obj.row_labels_)]
X_spco = X_spco[:, np.argsort(spco_obj.column_labels_)]
features_spco = feature_names.values[1:][np.argsort(spco_obj.column_labels_)]
df_spco = pd.DataFrame(data=X_spco, columns=features_spco)
plt.figure(figsize=(22,15))
sns.heatmap(df_spco, cmap=plt.cm.Blues)
plt.xlabel('Features', fontsize=20)
plt.ylabel('Observations', fontsize=20)

#%% Finding K Using Silhouette Coefficient

K_max = 15
k_vec = np.arange(2,K_max+1)
silh_coefs_km = np.zeros(len(k_vec),dtype=float)
silh_coefs_hc = np.zeros(len(k_vec),dtype=float)
silh_coefs_sc = np.zeros(len(k_vec),dtype=float)
print('\nFinding K Using Silhouette Coefficient')
for i in range(len(k_vec)):
    print('K = {} --- K-Means'.format(k_vec[i]))
    km_obj = KMeans(n_clusters=k_vec[i], init='k-means++', n_init=100, max_iter=500, random_state=5).fit(X_scaled)
    silh_coefs_km[i] = metrics.silhouette_score(X_scaled, km_obj.labels_, metric='euclidean')
    print('K = {} --- Hierarchical Clustering'.format(k_vec[i]))
    hc_obj = AgglomerativeClustering(n_clusters=k_vec[i], affinity='euclidean', linkage='ward').fit(X_scaled)
    silh_coefs_hc[i] = metrics.silhouette_score(X_scaled, hc_obj.labels_, metric='euclidean')
    print('K = {} --- Spectral Clustering'.format(k_vec[i]))
    sc_obj = SpectralClustering(n_clusters=k_vec[i], n_init=50, affinity='nearest_neighbors', random_state=5).fit(X_scaled)
    silh_coefs_sc[i] = metrics.silhouette_score(X_scaled, sc_obj.labels_, metric='euclidean')


#%%

plt.figure(figsize=(12,8))
plt.rc('font', size=16)
plt.plot(k_vec, silh_coefs_km, '-o', lw=3, markersize=10, label='K-Means')
plt.plot(k_vec, silh_coefs_hc, '-s', lw=3, markersize=10, label='Hierarchical')
plt.plot(k_vec, silh_coefs_sc, '->', lw=3, markersize=10, label='Spectral')
plt.legend(loc='upper right', fontsize=18)
plt.ylim([0.04,0.15])
plt.vlines(4,0.04,silh_coefs_km[2],linestyle='--',color='r')
plt.grid(axis='y',alpha=0.5)
plt.xticks(k_vec)
plt.xlabel('K (# of clusters)')
plt.ylabel('Silhouette Coefficient')


