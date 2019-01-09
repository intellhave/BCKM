import os
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KDTree
from scipy import io
from evaluation import *
from sklearn.decomposition import PCA, IncrementalPCA
import csv
import random
from sklearn.datasets import make_blobs
import pdb
from pandas import *

def ij_to_vectorized_idx(i,j,n):
    """    
    Returns the index of the vector generated from vectorizing matrix from its original i,j index
    [[A11, A12, A13],
     [A21, A22, A23],  --> [A11, A21, A31, A12, A22, A32, A13, A23, A33]
     [A31, A32, A33]  
    ]
    n: number of rows
    """
    return i+j*n

def print_matrix(x, n_cluster, N):
    M = x.reshape(N, n_cluster).T
    print DataFrame(M)
    

def is_binary(x):
    if abs(x)<1e-4: 
        return 0
    elif abs(x-1) < 1e-4:
        return 1
    else:
        return -1

def gen_gaussian_data(dimensions, n_clusters, point_per_cluster, std = 0.1, box_size = 5):
    X = None
    Y = None
    for i in range(n_clusters):    
        n_point_per_cluster = point_per_cluster[i]
        centroid = box_size*np.random.rand(dimensions)    
        X1, Y1 = make_blobs(n_point_per_cluster,              
                            n_features=dimensions, 
                            centers = centroid.reshape(1,-1))
        #pdb.set_trace()
        if X is None:
            X = X1
            Y = [0]*n_point_per_cluster
        else:
            X = np.concatenate((X, X1), axis = 0)
            Y += [i]*n_point_per_cluster

    return X, Y
        

def sample_must_link(X, labels, n_points):
    """
    Gerate sets of must-link constraints
    :param X: Set of data
    :param labels: set of ground truth labels
    """    
    must_link_constraints = []
    n_clusters = get_n_cluster(labels)
    # Start sampling:
    for cluster in range(n_clusters):
        pts_idx = [idx for idx, value in enumerate(labels) if value==cluster]
        sampled_idx = random.sample(pts_idx, n_points)
        #for i in range(n_points):
        i = 0
        for j in range(i+1, n_points):
            must_link_constraints+=[(sampled_idx[i], sampled_idx[j])]

    return must_link_constraints


def read_data(filename):
    """
    Read a csvfile and return X, Y where X is a matrix of features 
    and Y contains the groundtruth labels
    """
    X = []    
    with open(filename, 'r') as csvfile:
        data = csv.reader(csvfile, delimiter = ',')        
        for row in data:    
            X += [row]
    X = np.array(X).astype('float')
    labels = X[:,0].astype('int')
    #if filename != 'mnist_train':
    labels = labels - 1
    X = X[:,1:]
    return X, labels

def read_data_with_equal_classes(filename, n_classes, n_point_per_cluster=10):
    X, labels = read_data(filename)
    n_clusters = get_n_cluster(labels)
    cls_count = cluster_count(labels, n_clusters)
    if n_point_per_cluster < 0:
        n_point_per_cluster = min(cls_count)

    extracted_idx = []    
    for class_label in range(n_classes):
        pts_idx = [idx for (idx, value) in enumerate(labels) if value == class_label ]
        extracted_idx += [pts_idx[i] for i in range(n_point_per_cluster)]
        
    return X[extracted_idx,:], labels[extracted_idx] 


def read_BCLS_results(matfile):
    """
    Read results returned by BCLS:
    """
    data = io.loadmat(matfile)
    labels = data['ID'].ravel()
    runtime = data['runtime']
    return labels, runtime




def get_dataset_path(data_folder, dataset):
    return os.path.join(data_folder, dataset + ".csv")


def get_n_cluster(labels):
    unique_labels = np.unique(labels)
    labels = list(labels)
    unique_labels = list(unique_labels)
    n_clusters = len(unique_labels)
    return n_clusters


def cluster_count(label, n_cluster):
    count = []
    for i in range(n_cluster):
        cls_idx = [index for index,value in enumerate(label) if value == i]
        count += [len(cls_idx)]
    return count


def get_centroids_from_labels(X, labels):
    """
    From the dataset and current labelling, group points by labels and obtain the centroids
    """
    unique_labels = np.unique(labels)
    labels = list(labels)
    unique_labels = list(unique_labels)
    n_clusters = len(unique_labels)

    C = np.zeros((n_clusters, X.shape[1]))
    for c_idx, label in enumerate(unique_labels): 
        pts_idx = [idx for idx, value in enumerate(labels) if value == label]
        c = X[pts_idx,:]
        C[c_idx,:] = np.mean(c, axis = 0)

    return C

def scale_data(X):
    Xt = preprocessing.scale(X)    
    #Xt = preprocessing.minmax_scale(X)

    return Xt

def scale_data_minmax(X):
    #Xt = preprocessing.scale(X)    
    Xt = preprocessing.minmax_scale(X)

    return Xt

def pca_transform(X, n_components):
    pca = IncrementalPCA(n_components=n_components)
    pca.fit(X)
    Xt = pca.transform(X)
    return Xt




def balance_clusters(X, C, labels, pts_per_cluster):
    """
    From the initial centroids and labels provided by K-Means, try to balance the clusters 
    by reassiging the labels
    """
    unique_labels = np.unique(labels)
    labels = list(labels)
    unique_labels = list(unique_labels)
    n_clusters = len(unique_labels)

    # First, count the number of points in each cluters
    pts_count = []
    cluster_points = []

    for label in unique_labels:
        pts_idx = [idx for idx, value in enumerate(labels) if value == label]
        cluster_points += [pts_idx]
        pts_count += [len(pts_idx)]

    # Now, re-distribute the points. Assume that # cluster < 500, do bruteforce for now

    # First, find cluters that need more points
    missing_clusters = [idx for idx, count in enumerate(pts_count) if count < pts_per_cluster]
    
    # Pick points from redundant and "invite" them to missing clusters
    for cluster_idx in missing_clusters:        
        # Extract points belonging to this cluster             
        pts = cluster_points[cluster_idx]
        # Compute the number of missing points
        n_missing = pts_per_cluster  - len(pts)               
        for k in range(n_missing):
            # Look for clusters that has more points than necessary:
            redundant_clusters = [idx for idx, count in enumerate(pts_count) if count > pts_per_cluster]

            # Extract all points belonging to redundant sets
            pts_rd = []
            for r_cluster_idx in redundant_clusters:
                pts_rd += list(cluster_points[r_cluster_idx])    
            # Build a KD-Tree
            tree = KDTree(X[pts_rd,:])        
            #Obtain cluster centroid and query its nearest neighbors
            centroid = C[cluster_idx,:]
            _, q_idx = tree.query([centroid], 1)
            nn_idx = [pts_rd[idx] for idx in q_idx.ravel()]

            # Add nearest points to this cluster and remove them from their original cluster:
            for idx in nn_idx:
                # First, remove it from pts_rd:
                pts_rd.remove(idx)    
                
                #Remove itself from current cluster
                p_cluster = labels[idx]
                pts_count[p_cluster]-=1
                cls_pts = cluster_points[p_cluster]
                cls_pts.remove(idx)
                cluster_points[p_cluster] = cls_pts

                #Add this point to the missing cluster:
                cluster_points[cluster_idx] += [idx]
                pts_count[cluster_idx]+=1
                # Re-assign its label:
                labels[idx] = cluster_idx
            print pts_count[cluster_idx]
    print pts_count
    return labels


            
#Test 
if __name__ == '__main__':
    print 'This file contains utilities require for BCKMeasn'





    








