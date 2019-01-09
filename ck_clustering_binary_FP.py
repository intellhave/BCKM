from __future__ import division
import numpy as np
import time
from sklearn.cluster import KMeans
from numpy.linalg import inv
from scipy.linalg import norm
from tools import *
from evaluation import *
from cvxpy import *
import mosek
from qpsolvers import solve_qp
import scipy.sparse as sparse
import time
import sys
from scipy import io

def distortion (X, C, S):
    return np.linalg.norm(X- np.dot(C,S))

# Define a stream printer to grab output from MOSEK
def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()

def constrained_kmeans_binary(X, 
                              nClusters, 
                              n_pts_per_cluster,                         
                              link_constraints = None,
                              max_iter = 500, 
                              rho_start = 10.0,
                              init_labels = None,
                              split = False,
                              parallel = False, 
                              rho_inc_rate = 1.1, 
                              centroid_convergence =1e-1, 
                              binary_convergence_thres = 1e-1,
                              splitting_convergence = 1e-1,
                              cvx_solver = MOSEK):
    """
    performs K-means clustering using binary optimization
    input: 
    - X: data, ndarray of dxN where d is dimension and N is the number of data instances
    - nClusters: Number of clusters to segment.
    - L: Previous labels
    """

    N = X.shape[1]
    labels = init_labels
    
    #Use K-Means to initialize the labels and centroids if initialization is not provided
    if init_labels is None:
        kmeans = KMeans(n_clusters = nClusters, 
                        init='k-means++',
                        n_init = 10,
                        max_iter = 10,
                        )        
        kmeans.fit(X.T)
        labels = kmeans.labels_    
        
    #Re-compute centroids
    print 'Points distribution after initialization:', cluster_count(labels, nClusters)

    # Obtain initial centroids from label initialization, then initialize the assignment matrix:    
    C0 = get_centroids_from_labels(X.T, labels).T                                
    S0 = assignment_matrix_from_labels(labels)

    # Start the iterations 
    converged = False
    iter = 0
    C = C0
    S = S0    
    
    obj = distortion(X, C, S)
    print 'Initial Distortion = ', obj

    while (not converged and iter < max_iter):       
        # Optimize C:
        prev_C = C        
        C = cluster_assignment(X, S, C)
        obj = distortion(X, C, S)
        print 'New distortion = ', obj

        # Assign labels based on the current centroids
        labels, S = binary_assignment(X, 
                                   C, 
                                   n_pts_per_cluster, 
                                   S, 
                                   rho_start, 
                                   link_constraints=link_constraints,
                                   split=split,
                                   parallel=parallel,
                                   rho_inc_rate=rho_inc_rate, 
                                   convergence_thres=binary_convergence_thres, 
                                   splitting_convergence = splitting_convergence,
                                   cvx_solver=cvx_solver)

        #print S
        #print cluster_count(labels, nClusters) 
        
        obj = distortion(X, C, S)
        print 'Distortion after assignment = ', obj      
        
        centroid_gap = np.linalg.norm(prev_C - C)        
        #print centroid_gap
        if centroid_gap<centroid_convergence:
            converged = True
        iter += 1

    C = np.asarray(C)
    return labels, C
        

def cluster_assignment(X, S, C0, r_lambda = 0.00001):
    """ solve min_C || X - CS ||^2
    This is done by solving min_C || X - CS||^2_F + lambda* ||C||^2_F,
    which has the closed form solution as
                     C  = XS'(SS' + lambda*I)^-1
    Parameters 
    ----------
    X : input data
    S : assignment matrix
    C0: initial solution
    """
    print('----------Finding clusters using regularized least square-------------')   
    B1 = np.dot(S, S.T)
    B1 = B1 + r_lambda*(np.eye(B1.shape[0]))
    B1 = inv(B1)
    B2 = np.dot(X, S.T)
    C = np.dot(B2, B1)
    return C

def binary_assignment(X, 
                      C, 
                      pts_per_cluster, 
                      S0, 
                      rho, 
                      link_constraints = None,
                      split = False,
                      parallel = False,
                      maxIter = 5000, 
                      convergence_thres = 1e-2, 
                      splitting_convergence=1e-1,
                      rho_inc_rate = 1.1, 
                      cvx_solver = MOSEK):
    """    
    Solve the optimization problem with binary constraints
    min_{S} || X - CS ||^2_F 
    This is implemented with augmented Lagranginan method
    """    

    print('--------------Optimizing Binary Assignments from clusters--------')
    N = X.shape[1]
    d = X.shape[0]
    nClusters = C.shape[1]    

    St = S0    
    Vt = np.round(St)
    prev_S = St
    # Prepare the rho matrices
    rho_plus = rho*np.ones((nClusters, N))
    rho_minus = rho*np.ones((nClusters, N))
    """
    Now, start solving min ||X - CS|| + rho*(<S, V> - nClusters*N)
                       s.t. -1 <= S <= 1
                            || V ||^2_F <= KN 
    """      
    # Start a MOSEK environment
    with mosek.Env() as env:     
        with env.Task(0,0) as task:                 
            # task.set_Stream(mosek.streamtype.log, streamprinter) # Uncomment this to monitor the MOSEK solver
        
            # Define the number of variables and constraints:
            numvar = nClusters * N      # Variables for the matrix S                        

            task.appendvars(numvar*3)   # The variables comprise of variables for S and D
            task.putobjsense(mosek.objsense.minimize)
            """ 
                Prepare objective function for S (Prepare the linear matrix first)
            """             
            XC = np.zeros(shape=(nClusters, N))
            for i in range(nClusters):
                for j in range(N):
                    XC[i,j] = np.linalg.norm(X[:, j] - C[:, i])**2           
            XC = XC
            
            """
                Enforcing constraints
            """    
            # Bound Constraints                        
            for i in range(numvar): # 0 <= S <=1                
                task.putvarbound(i, mosek.boundkey.ra, 0, 1.0)                
            for i in range(numvar, 3*numvar):
                task.putvarbound(i, mosek.boundkey.lo, 0.0, 1e10)                

            # Each point is assigned to one cluster             
            n_constraints = -1
            for i in range(N):                        
                n_constraints += 1                        
                task.appendcons(1)
                var_idx = [ij_to_vectorized_idx(j, i, nClusters) for j in range(nClusters)]
                task.putarow(n_constraints, var_idx, [1]*nClusters)                                        
                task.putconbound(n_constraints, mosek.boundkey.fx, 1.0, 1.0)

            # Distribute points into clusters
            for i in range(nClusters):                                
                n_constraints += 1
                task.appendcons(1) 
                var_idx = [ij_to_vectorized_idx(i, j, nClusters) for j in range(N)]
                task.putarow(n_constraints, var_idx, [1]*N)                                                        
                task.putconbound(n_constraints, mosek.boundkey.lo, pts_per_cluster , 1e20)                    

            # If link constraint is provided, enforce it:
            if link_constraints is not None:
                for constraint in link_constraints:
                    p = constraint[0]
                    q = constraint[1]
                    for j in range(nClusters):
                        n_constraints +=1
                        task.appendcons(1) 
                        var_idx = [ij_to_vectorized_idx(j, p, nClusters), 
                                    ij_to_vectorized_idx(j, q, nClusters)]
                        task.putarow(n_constraints, var_idx,[1, -1])
                        task.putconbound(n_constraints, mosek.boundkey.fx, 0.0, 0.0)

            # Constraints for D variable: dij+ >= Sij - Vij and dij- >= Vij - Sij            
            d_constraints_start_idx = n_constraints
            for j in range(N):
                for i in range(nClusters):
                    idx = ij_to_vectorized_idx(i,j,nClusters)
                    n_constraints +=1
                    task.appendcons(1)
                    var_idx = [idx, numvar+idx]
                    task.putarow(n_constraints, var_idx, [-1, 1]) # -xi + di+ >= yi
                    
                    n_constraints+=1
                    task.appendcons(1)
                    var_idx = [idx, 2*numvar + idx]
                    task.putarow(n_constraints, var_idx, [1, 1]) # xi + di- >= yi

            # Start iterations        
            """
                Solve S
            """    
            converged = False
            it = 0    
            alpha = 0.5
            scale_factor = 1
            
            while not converged and it < maxIter:                           
                
                print('--Solving S')
                prev_S = St        
                prev_V = Vt        
                # Prepare objective 
                cij = np.zeros(shape=(3*numvar,))
                for j in range(N):
                    for i in range(nClusters):                        
                        cij[ij_to_vectorized_idx(i,j, nClusters)] = alpha*scale_factor*XC[i,j]
                                
                if Vt is not None:
                    # Put cij for d variables
                    for j in range(N):
                        for i in range(nClusters):
                            idx = ij_to_vectorized_idx(i,j, nClusters)
                            cij[idx+numvar] = (1-alpha)*rho_plus[i,j]       
                            cij[idx+2*numvar] = (1-alpha)*rho_minus[i,j]       
                                
                          
                    # Incorporate pentalty cost functions
                    constr_idx = d_constraints_start_idx
                    for j in range(N):
                        for i in range(nClusters):
                            idx = ij_to_vectorized_idx(i, j, nClusters)
                            constr_idx +=1
                            task.putconbound(constr_idx, mosek.boundkey.lo, -Vt[i,j], 1e10)
                            constr_idx+=1
                            task.putconbound(constr_idx, mosek.boundkey.lo, Vt[i,j], 1e10)
                                
                for i in range(numvar*3):
                    task.putcj(i, cij[i])    
                # Solve the problem
                task.optimize()

                #Extract the solutions
                xx = [0.]*numvar*3
                task.getxx(mosek.soltype.bas, xx)
                d_plus = np.asarray(xx[numvar:2*numvar], dtype='float')
                d_minus = np.asarray(xx[2*numvar:3*numvar], dtype='float')
                xx = xx[0:numvar]
                
                solsta = task.getsolsta(mosek.soltype.bas)            
                xx = np.asarray(xx, dtype='float')                                   
                St = xx.reshape(N,nClusters).T
                                                                  
                # Update Vt and rho 
                print '--Update V and RHO'
                if Vt is None:
                    Vt = np.round(St)
                for j in range(N):
                    for i in range(nClusters):                        
                        if rho_plus[i,j]*(1-St[i,j]) <= rho_minus[i,j]*(St[i,j]):
                            Vt[i,j] = 1
                            rho_plus[i,j] = rho_plus[i,j]*rho_inc_rate
                        else:
                            Vt[i,j] = 0 
                            rho_minus[i,j] = rho_minus[i,j]*rho_inc_rate


                #rho = rho*1.5                                
                it += 1            
                sv_diff = np.linalg.norm(prev_S - St)**2 +  np.linalg.norm(prev_V - Vt)**2                 
                if sv_diff < convergence_thres:
                    converged = True


    # Get labels from St:
    labels = []
    for i in range(N):
        labels.append(np.argmax(St[:,i]))    
    
    # Convert solution back to {0,1} domain
    S = 0.5*(Vt + np.ones(Vt.shape))
    return labels, S
 

def assignment_matrix_from_labels(labels):
    """converts set of labels into assignment matrix
    
    Parameters
    ----------
    labels : a list of labels    

    Attributes
    ----------
    S: Assignment matrix obtained from label

    """
    n_points = len(labels)
    unique_labels = set(labels)
    n_lables = len(unique_labels)
    S = -1*np.ones(shape=(n_lables, n_points), dtype='int')

    for j in range(n_points):
        S[labels[j], j] = 1

    return S


