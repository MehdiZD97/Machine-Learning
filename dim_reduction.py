# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 12:07:38 2022

@author: mz52
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, NMF, FastICA, KernelPCA
from sklearn.manifold import TSNE, SpectralEmbedding, MDS
from sklearn.cluster import SpectralBiclustering, SpectralCoclustering
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

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


#%% PCA

print("\nApplying PCA")
pca_obj = PCA(n_components=0.90)
X_pca = pca_obj.fit_transform(X_scaled)
K_pca = X_pca.shape[1]
print("New dimensions: ", X_pca.shape)

#%% Kernel PCA

X_pca_kernel_mat = np.zeros((X_scaled.shape[0],2,2))
degrees = [2,3]
for i in range(2):
    kernel_pca_obj = KernelPCA(n_components=2, kernel='poly', degree=degrees[i], gamma=None)
    X_pca_kernel_mat[:,:,i] = kernel_pca_obj.fit_transform(X_scaled)

plt.figure(figsize=(16,6))
plt.rc('font', size=14)
plt.subplot(1,2,1)
scatter = plt.scatter(X_pca_kernel_mat[:,0,0], X_pca_kernel_mat[:,1,0], c=auth_label)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Polynomial Kernel with degree = {}'.format(degrees[0]))
plt.subplot(1,2,2)
scatter = plt.scatter(X_pca_kernel_mat[:,0,1], X_pca_kernel_mat[:,1,1], c=auth_label)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Polynomial Kernel with degree = {}'.format(degrees[1]))
legend_elems = ['Austen', 'London', 'Milton', 'Shakespeare']
plt.legend(handles=scatter.legend_elements()[0], labels=legend_elems, fontsize=18, bbox_to_anchor=(0.7,-0.15+2.55-2.55), ncol=4, fancybox=True, shadow=True)


X_pca_kernel_mat = np.zeros((X_scaled.shape[0],2,2))
kernels=['sigmoid', 'cosine']
for i in range(2):
    kernel_pca_obj = KernelPCA(n_components=2, kernel=kernels[i])
    X_pca_kernel_mat[:,:,i] = kernel_pca_obj.fit_transform(X_scaled)

plt.figure(figsize=(16,6))
plt.rc('font', size=14)
plt.subplot(1,2,1)
scatter = plt.scatter(X_pca_kernel_mat[:,0,0], X_pca_kernel_mat[:,1,0], c=auth_label)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('{} Kernel PCA'.format(kernels[0]))
plt.subplot(1,2,2)
scatter = plt.scatter(X_pca_kernel_mat[:,0,1], X_pca_kernel_mat[:,1,1], c=auth_label)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('{} Kernel PCA'.format(kernels[1]))
legend_elems = ['Austen', 'London', 'Milton', 'Shakespeare']
plt.legend(handles=scatter.legend_elements()[0], labels=legend_elems, fontsize=18, bbox_to_anchor=(0.7,-0.15+2.55-2.55), ncol=4, fancybox=True, shadow=True)




#%% PCA Visualization for Observations

K_pca_pairplot = 5
print("Visualizing PCA results")

pca_var = pca_obj.explained_variance_
plt.figure(figsize=(16,6))
plt.rc('font', size=14)
plt.subplot(1,2,1)
plt.plot(np.arange(K_pca)+1, pca_var/100, lw=3)
plt.xticks(np.arange(1,K_pca+1,4))
plt.xlabel('# of PCs')
plt.grid(axis='y', alpha=0.5)
plt.ylabel('Variance (%)')
plt.subplot(1,2,2)
plt.plot(np.arange(K_pca)+1, pca_var.cumsum()/100, lw=3)
plt.xticks(np.arange(1,K_pca+1,4))
plt.xlabel('# of PCs')
plt.ylim([0,1])
plt.grid(axis='y', alpha=0.5)
plt.ylabel('Cumulative Variance (%)')

plt.figure(figsize=(8,6))
plt.rc('font', size=14)
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=auth_label)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
legend_elems = ['Austen', 'London', 'Milton', 'Shakespeare']
plt.legend(handles=scatter.legend_elements()[0], labels=legend_elems, fontsize=12)

if K_pca>4:
    n_figs = 4
    plt.figure(figsize=(16,12))
    plt.rc('font', size=14)
    for i in range(4):
        plt.subplot(2,2,i+1)
        scatter = plt.scatter(X_pca[:,0], X_pca[:,i+1], c=auth_label)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component {}'.format(i+2))
        legend_elems = ['Austen', 'London', 'Milton', 'Shakespeare']
    plt.legend(handles=scatter.legend_elements()[0], labels=legend_elems, fontsize=18, bbox_to_anchor=(0.7,-0.15+2.55-2.55), ncol=4, fancybox=True, shadow=True)

if K_pca > K_pca_pairplot-1:
    df_pca = pd.DataFrame(data=X_pca[:,0:K_pca_pairplot], columns=['PC{}'.format(i+1) for i in range(K_pca_pairplot)])
    df_pca = df_pca.merge(auth, left_index=True, right_index=True)
    sns.pairplot(df_pca, hue='Name')

#%% PCA Visualization for Features

components_pca = pca_obj.components_.T

plt.figure(figsize=(14,5))
plt.rc('font', size=12)
plt.subplot(1,2,1)
plt.stem(components_pca[:,0], basefmt='C3-')
plt.grid(axis='both', alpha=0.5)
plt.xlabel('Feature Index')
plt.ylabel('V1 (Coefficients for PC1)')
plt.subplot(1,2,2)
plt.stem(components_pca[:,1])
plt.grid(axis='both', alpha=0.5)
plt.xlabel('Feature Index')
plt.ylabel('V2 (Coefficients for PC2)')

plt.figure(figsize=(10,16))
df_pca_comps = pd.DataFrame(data=components_pca[:,0:4], columns=['PC{}'.format(i+1) for i in range(4)], index=feature_names.values[1:])
sns.heatmap(df_pca_comps)
plt.ylabel('Features', fontsize=16)

plt.figure(figsize=(12,10))
plt.rc('font', size=14)
plt.scatter(components_pca[:,0], components_pca[:,1], s=30, c='r')
plt.xlim([-0.25,0.3])
plt.ylim([-0.25,0.3])
plt.axvline(x=0.15, linestyle='--')
plt.axhline(y=0.15, linestyle='--')
plt.axhspan(0.15,0.3, facecolor='b', alpha=0.2)
plt.axvspan(0.15,0.3, facecolor='b', alpha=0.2)
for i in range(components_pca.shape[0]):
    plt.annotate(feature_names[i+1], (components_pca[i,0]-0.005,components_pca[i,1]+0.005))
plt.xlabel('V1 (Coefficients for PC1)')
plt.ylabel('V2 (Coefficients for PC2)')

#%% NMF

print("\nApplying NMF")
K_nmf = 5
nmf_obj = NMF(n_components=K_nmf, init='random', alpha_W=0, alpha_H=0, l1_ratio=0, max_iter=2000)
X_nmf = nmf_obj.fit_transform(X)
print("New dimensions: ", X_nmf.shape)

#%% NMF Visualization for Observations

K_nmf_pairplot = 5
print("Visualizing NMF results")
plt.figure(figsize=(8,6))
plt.rc('font', size=14)
scatter = plt.scatter(X_nmf[:,0], X_nmf[:,1], c=auth_label)
plt.xlabel('NMF Component 1')
plt.ylabel('NMF Component 2')
legend_elems = ['Austen', 'London', 'Milton', 'Shakespeare']
plt.legend(handles=scatter.legend_elements()[0], labels=legend_elems, fontsize=12)

if K_nmf>4:
    n_figs = 4
    plt.figure(figsize=(16,12))
    plt.rc('font', size=14)
    for i in range(n_figs):
        plt.subplot(2,2,i+1)
        scatter = plt.scatter(X_nmf[:,0], X_nmf[:,i+1], c=auth_label)
        plt.xlabel('NMF Component 1')
        plt.ylabel('NMF Component {}'.format(i+2))
        legend_elems = ['Austen', 'London', 'Milton', 'Shakespeare']
    plt.legend(handles=scatter.legend_elements()[0], labels=legend_elems, fontsize=18, bbox_to_anchor=(0.7,-0.15+2.55-2.55), ncol=4, fancybox=True, shadow=True)

if K_nmf > K_nmf_pairplot-1:
    df_nmf = pd.DataFrame(data=X_nmf[:,0:K_nmf_pairplot], columns=['NMF{}'.format(i+1) for i in range(K_nmf_pairplot)])
    df_nmf = df_nmf.merge(auth, left_index=True, right_index=True)
    sns.pairplot(df_nmf, hue='Name')

#%% NMF Visualization for Features

components_nmf = nmf_obj.components_.T

plt.figure(figsize=(14,5))
plt.rc('font', size=12)
plt.subplot(1,2,1)
plt.stem(components_nmf[:,0])
plt.grid(axis='both', alpha=0.5)
plt.xlabel('Feature Index')
plt.ylabel('NMF1 (Coefficients)')
plt.subplot(1,2,2)
plt.stem(components_nmf[:,1])
plt.grid(axis='both', alpha=0.5)
plt.xlabel('Feature Index')
plt.ylabel('NMF2 (Coefficients)')

plt.figure(figsize=(10,16))
df_nmf_comps = pd.DataFrame(data=components_nmf[:,0:K_nmf], columns=['NMF{}'.format(i+1) for i in range(K_nmf)], index=feature_names.values[1:])
sns.heatmap(df_nmf_comps)
plt.ylabel('Features', fontsize=16)

# Dimensions should be adjusted 
plt.figure(figsize=(12,10))
plt.rc('font', size=14)
plt.scatter(components_nmf[:,0], components_nmf[:,1], s=30, c='r')
plt.xlim([-1,18.5])
plt.ylim([-0.5,7])
plt.axvline(x=5, linestyle='--')
plt.axhline(y=3, linestyle='--')
plt.axhspan(3,7, facecolor='b', alpha=0.2)
plt.axvspan(5,18.5, facecolor='b', alpha=0.2)
for i in range(components_nmf.shape[0]):
    plt.annotate(feature_names[i+1], (components_nmf[i,0]-0.005,components_nmf[i,1]+0.005))
plt.xlabel('NMF1 (Coefficients)')
plt.ylabel('NMF2 (Coefficients)')

#%% tSNE

print("\nApplying tSNE")
K_tsne = 3 # for 'barnes_hut' must be less than 4
tsne_obj = TSNE(n_components=K_tsne, perplexity=30.0, learning_rate=200, n_iter=1000, init='pca', method='exact')  # method = 'exact' is slower but more accurate than 'barnes_hut'
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    X_tsne = tsne_obj.fit_transform(X_scaled)
print("New dimensions: ", X_tsne.shape)
print("KL-Divergence: ", tsne_obj.kl_divergence_)

#%% tSNE Visualization for Observations

K_tsne_pairplot = 3
print("Visualizing tSNE results")
plt.figure(figsize=(8,6))
plt.rc('font', size=14)
scatter = plt.scatter(X_tsne[:,0], X_tsne[:,1], c=auth_label)
plt.xlabel('tSNE Component 1')
plt.ylabel('tSNE Component 2')
legend_elems = ['Austen', 'London', 'Milton', 'Shakespeare']
plt.legend(handles=scatter.legend_elements()[0], labels=legend_elems, fontsize=12)

if K_tsne>4:
    n_figs = 4
    plt.figure(figsize=(16,12))
    plt.rc('font', size=14)
    for i in range(n_figs):
        plt.subplot(2,2,i+1)
        scatter = plt.scatter(X_tsne[:,0], X_tsne[:,i+1], c=auth_label)
        plt.xlabel('tSNE Component 1')
        plt.ylabel('tSNE Component {}'.format(i+2))
        legend_elems = ['Austen', 'London', 'Milton', 'Shakespeare']
    plt.legend(handles=scatter.legend_elements()[0], labels=legend_elems, fontsize=18, bbox_to_anchor=(0.7,-0.15+2.55-2.55), ncol=4, fancybox=True, shadow=True)

if K_tsne > K_tsne_pairplot-1:
    df_tsne = pd.DataFrame(data=X_tsne[:,0:K_tsne_pairplot], columns=['tSNE{}'.format(i+1) for i in range(K_tsne_pairplot)])
    df_tsne = df_tsne.merge(auth, left_index=True, right_index=True)
    sns.pairplot(df_tsne, hue='Name')
    

#%% UMAP

print("\nApplying UMAP")
K_umap = 5 # anything b/w 2 and 100
umap_obj = umap.umap_.UMAP(n_neighbors=15, n_components=K_umap, metric='euclidean', init='spectral') # metric='euclidean'/'manhattan'/'chebyshev' & init='spectral'/'pca'/'random'
X_umap = umap_obj.fit_transform(X_scaled)
print("New dimensions: ", X_umap.shape)

#%% UMAP Visualization

K_umap_pairplot = 5
print("Visualizing UMAP results")
plt.figure(figsize=(8,6))
plt.rc('font', size=14)
scatter = plt.scatter(X_umap[:,0], X_umap[:,1], c=auth_label)
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
legend_elems = ['Austen', 'London', 'Milton', 'Shakespeare']
plt.legend(handles=scatter.legend_elements()[0], labels=legend_elems, fontsize=12)

if K_umap>4:
    n_figs = 4
    plt.figure(figsize=(16,12))
    plt.rc('font', size=14)
    for i in range(n_figs):
        plt.subplot(2,2,i+1)
        scatter = plt.scatter(X_umap[:,0], X_umap[:,i+1], c=auth_label)
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component {}'.format(i+2))
        legend_elems = ['Austen', 'London', 'Milton', 'Shakespeare']
    plt.legend(handles=scatter.legend_elements()[0], labels=legend_elems, fontsize=18, bbox_to_anchor=(0.7,-0.15+2.55-2.55), ncol=4, fancybox=True, shadow=True)

if K_umap > K_umap_pairplot-1:
    df_umap = pd.DataFrame(data=X_umap[:,0:K_umap_pairplot], columns=['UMAP{}'.format(i+1) for i in range(K_umap_pairplot)])
    df_umap = df_umap.merge(auth, left_index=True, right_index=True)
    sns.pairplot(df_umap, hue='Name')


#%% Spectral Embedding

print("\nApplying Spectral Embedding")
K_se = 5
se_obj = SpectralEmbedding(n_components=K_se, affinity='nearest_neighbors', gamma=None) # affinity='nearest_neighbors'/'rbf'
X_se = se_obj.fit_transform(X_scaled)
print("New dimensions: ", X_se.shape)


#%% SE Visualization

K_se_pairplot = 5
print("Visualizing Spectral Embedding results")
plt.figure(figsize=(8,6))
plt.rc('font', size=14)
scatter = plt.scatter(X_se[:,0], X_se[:,1], c=auth_label)
plt.xlabel('SE Component 1')
plt.ylabel('SE Component 2')
legend_elems = ['Austen', 'London', 'Milton', 'Shakespeare']
plt.legend(handles=scatter.legend_elements()[0], labels=legend_elems, fontsize=12)

se_mat = se_obj.affinity_matrix_.toarray()
plt.figure(figsize=(10,8))
sns.heatmap(se_mat, cmap=plt.cm.Blues)
plt.xlabel('Observation Index')
plt.ylabel('Observation Index')

if K_se>4:
    n_figs = 4
    plt.figure(figsize=(16,12))
    plt.rc('font', size=14)
    for i in range(n_figs):
        plt.subplot(2,2,i+1)
        scatter = plt.scatter(X_se[:,0], X_se[:,i+1], c=auth_label)
        plt.xlabel('SE Component 1')
        plt.ylabel('SE Component {}'.format(i+2))
        legend_elems = ['Austen', 'London', 'Milton', 'Shakespeare']
    plt.legend(handles=scatter.legend_elements()[0], labels=legend_elems, fontsize=18, bbox_to_anchor=(0.7,-0.15+2.55-2.55), ncol=4, fancybox=True, shadow=True)

if K_se > K_se_pairplot-1:
    df_se = pd.DataFrame(data=X_se[:,0:K_se_pairplot], columns=['SE{}'.format(i+1) for i in range(K_se_pairplot)])
    df_se = df_se.merge(auth, left_index=True, right_index=True)
    sns.pairplot(df_se, hue='Name')

#%% MDS

print("\nApplying MDS")
K_mds = 5
mds_obj = MDS(n_components=K_mds, max_iter=500, dissimilarity='euclidean') # dissimilarity='euclidean'/'precomputed'
X_mds = mds_obj.fit_transform(X_scaled)
print("Distance Metric: Euclidean")
print("New dimensions: ", X_mds.shape)

#%% MDS Visualization

K_mds_pairplot = 5
print("Visualizing MDS results")
plt.figure(figsize=(8,6))
plt.rc('font', size=14)
scatter = plt.scatter(X_mds[:,0], X_mds[:,1], c=auth_label)
plt.xlabel('MDS Component 1')
plt.ylabel('MDS Component 2')
legend_elems = ['Austen', 'London', 'Milton', 'Shakespeare']
plt.legend(handles=scatter.legend_elements()[0], labels=legend_elems, fontsize=12)

mds_dissim_mat = mds_obj.dissimilarity_matrix_
plt.figure(figsize=(10,8))
sns.heatmap(mds_dissim_mat, cmap=plt.cm.Blues)
plt.xlabel('Observation Index')
plt.ylabel('Observation Index')

if K_mds>4:
    n_figs = 4
    plt.figure(figsize=(16,12))
    plt.rc('font', size=14)
    for i in range(n_figs):
        plt.subplot(2,2,i+1)
        scatter = plt.scatter(X_mds[:,0], X_mds[:,i+1], c=auth_label)
        plt.xlabel('MDS Component 1')
        plt.ylabel('MDS Component {}'.format(i+2))
        legend_elems = ['Austen', 'London', 'Milton', 'Shakespeare']
    plt.legend(handles=scatter.legend_elements()[0], labels=legend_elems, fontsize=18, bbox_to_anchor=(0.7,-0.15+2.55-2.55), ncol=4, fancybox=True, shadow=True)

if K_mds > K_mds_pairplot-1:
    df_mds = pd.DataFrame(data=X_mds[:,0:K_mds_pairplot], columns=['MDS{}'.format(i+1) for i in range(K_mds_pairplot)])
    df_mds = df_mds.merge(auth, left_index=True, right_index=True)
    sns.pairplot(df_mds, hue='Name')


#%% ICA

print("\nApplying ICA")
K_ica = 5
ica_obj = FastICA(n_components=K_ica, max_iter=400)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    X_ica = ica_obj.fit_transform(X_scaled)
print("New dimensions: ", X_ica.shape)

#%% ICA Visualization for Observations

K_ica_pairplot = 5
print("Visualizing ICA results")
plt.figure(figsize=(8,6))
plt.rc('font', size=14)
scatter = plt.scatter(X_ica[:,0], X_ica[:,1], c=auth_label)
plt.xlabel('ICA Component 1')
plt.ylabel('ICA Component 2')
legend_elems = ['Austen', 'London', 'Milton', 'Shakespeare']
plt.legend(handles=scatter.legend_elements()[0], labels=legend_elems, fontsize=12)

if K_ica>4:
    n_figs = 4
    plt.figure(figsize=(16,12))
    plt.rc('font', size=14)
    for i in range(n_figs):
        plt.subplot(2,2,i+1)
        scatter = plt.scatter(X_ica[:,0], X_ica[:,i+1], c=auth_label)
        plt.xlabel('ICA Component 1')
        plt.ylabel('ICA Component {}'.format(i+2))
        legend_elems = ['Austen', 'London', 'Milton', 'Shakespeare']
    plt.legend(handles=scatter.legend_elements()[0], labels=legend_elems, fontsize=18, bbox_to_anchor=(0.7,-0.15+2.55-2.55), ncol=4, fancybox=True, shadow=True)

if K_ica > K_ica_pairplot-1:
    df_ica = pd.DataFrame(data=X_ica[:,0:K_ica_pairplot], columns=['ICA{}'.format(i+1) for i in range(K_ica_pairplot)])
    df_ica = df_ica.merge(auth, left_index=True, right_index=True)
    sns.pairplot(df_ica, hue='Name')
    
#%% ICA Visualization for Features

components_ica = ica_obj.components_.T

plt.figure(figsize=(16,5))
plt.rc('font', size=12)
plt.subplot(1,2,1)
plt.stem(components_ica[:,0])
plt.grid(axis='both', alpha=0.5)
plt.xlabel('Feature Index')
plt.ylabel('ICA1 (Coefficients)')
plt.subplot(1,2,2)
plt.stem(components_ica[:,1])
plt.grid(axis='both', alpha=0.5)
plt.xlabel('Feature Index')
plt.ylabel('ICA2 (Coefficients)')

plt.figure(figsize=(10,16))
df_ica_comps = pd.DataFrame(data=components_ica[:,0:K_ica], columns=['ICA{}'.format(i+1) for i in range(K_ica)], index=feature_names.values[1:])
sns.heatmap(df_ica_comps)
plt.ylabel('Features', fontsize=16)

plt.figure(figsize=(12,8))
plt.rc('font', size=14)
plt.scatter(components_ica[:,0], components_ica[:,1], s=30, c='r')
plt.xlim([-0.0045,0.006])
plt.ylim([-0.0082,0.0065])
plt.axvline(x=0.002, linestyle='--')
plt.axhline(y=0.002, linestyle='--')
plt.axhspan(0.002,0.0065, facecolor='b', alpha=0.2)
plt.axvspan(0.002,0.006, facecolor='b', alpha=0.2)
for i in range(components_ica.shape[0]):
    plt.annotate(feature_names[i+1], (components_ica[i,0]-0.00005,components_ica[i,1]+0.0001))
plt.xlabel('ICA1 (Coefficients)')
plt.ylabel('ICA2 (Coefficients)')

#%% Spectral Biclustering

K_spbi = 4
print("\nApplying Spectral Bi-clustering")
X_clust = X_scaled
spbi_obj = SpectralBiclustering(n_clusters=K_spbi, method='bistochastic', init='k-means++') # method = 'bistochastic'/'scale'/'log'
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

#%% Spectral Co-clustering

K_spco = 4
print("\nApplying Spectral Co-clustering")
X_clust = X_scaled
spco_obj = SpectralCoclustering(n_clusters=4)
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

