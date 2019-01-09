# Place holder for synthetic experiments
from __future__ import division
import numpy as np
from sklearn import preprocessing
from evaluation import mutual_info
from tools import gen_gaussian_data, sample_must_link

#from constrained_clustering import constrained_kmeans_binary
from ck_clustering_binary_FP import constrained_kmeans_binary
import time

n = 1000
n_clusters = 50
dimension = 512
n_pts_per_cluster = np.floor(n/n_clusters).astype('int')

random_state = 0

# Generate
pts_per_cluster = [n_pts_per_cluster]*n_clusters
X, y = gen_gaussian_data(dimension, 
                        n_clusters, 
                         pts_per_cluster,
                         box_size=0.1,
                         std = 0.1)

n_sampled_points = np.int(np.ceil(n_pts_per_cluster*0.6))
must_link = sample_must_link(X, y, n_sampled_points)

truth_labels = y
scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)

# Constraint K_means
start_time = time.time()

bck_pred, _ = constrained_kmeans_binary(X.T, n_clusters, 
                                      n_pts_per_cluster,
                                      link_constraints=must_link,
                                      init_labels=None,
                                      centroid_convergence=1e-4,
                                      binary_convergence_thres=1e-4,
                                      rho_start=0.5,
                                      rho_inc_rate=1.01,
                                      split=False,
                                      parallel=False,
                                      splitting_convergence=1e-1
                                      #cvx_solver='MOSEK'
                                      )
bck_time = time.time() - start_time

# Report 
bck_nmi = mutual_info(bck_pred, y)
print('BCKM nmi: {0:.3f}'.format(bck_nmi))
print('BCKM Time: {0:.3f}'.format(bck_time))


