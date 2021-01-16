'''
hw0.py
Author: Zoe Hsieh
Tufts COMP 135 Intro ML, Fall 2020
Summary
-------
Complete the problems below to demonstrate your mastery of numerical Python.
Submit to the autograder to be evaluated.
You can do a basic check of the doctests via:
$ python -m doctest hw0.py
'''

import numpy as np
import random
import math


def split_into_train_and_test(x_all_LF, frac_test=0.5, random_state=None):
    ''' Divide provided array into train and test sets along first dimension
    User can provide random number generator object to ensure reproducibility.
    Args
    ----
    x_all_LF : 2D np.array, shape = (n_total_examples, n_features) (L, F)
        Each row is a feature vector
    frac_test : float, fraction between 0 and 1
        Indicates fraction of all L examples to allocate to the "test" set
        Returned test set will round UP if frac_test * L is not an integer.
        e.g. if L = 10 and frac_test = 0.31, then test set has N=4 examples
    random_state : np.random.RandomState instance or integer or None
        If int, will create RandomState instance with provided value as seed
        If None, defaults to current numpy random number generator np.random.
    Returns
    -------
    x_train_MF : 2D np.array, shape = (n_train_examples, n_features) (M, F)
        Each row is a feature vector
        Should be a separately allocated array, NOT a view of any input array
    x_test_NF : 2D np.array, shape = (n_test_examples, n_features) (N, F)
        Each row is a feature vector
        Should be a separately allocated array, NOT a view of any input array
    Post Condition
    --------------
    This function should be side-effect free. Provided input array x_all_LF
    should not change at all (not be shuffled, etc.)
    Examples
    --------
    >>> x_LF = np.eye(10)
    >>> xcopy_LF = x_LF.copy() # preserve what input was before the call
    >>> train_MF, test_NF = split_into_train_and_test(
    ...     x_LF, frac_test=0.201, random_state=np.random.RandomState(0))
    >>> train_MF.shape
    (7, 10)
    >>> test_NF.shape
    (3, 10)
    >>> print(train_MF)
    [[0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
     [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]]
    >>> print(test_NF)
    [[0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]]
    ## Verify that input array did not change due to function call
    >>> np.allclose(x_LF, xcopy_LF)
    True
    References
    ----------
    For more about RandomState, see:
    https://stackoverflow.com/questions/28064634/random-state-pseudo-random-numberin-scikit-learn
    '''
    if random_state is None:
        random_state = np.random
 
    #random_state = np.random.RandomState(0)
    #rdm = np.random.mtrand.RandomState(random_state)       
    #index_1 = rdm.choice(mushroom_data.shape[0],test_size,replace=False)

    L,F = x_all_LF.shape
    #split data up by frac
    copy_x_all = np.zeros((L,F))
    copy_x_all = x_all_LF.copy()

    #shuffle copy
    #np.random.RandomState(int(random_state)).shuffle(copy_x_all)    
    np.random.RandomState(random_state).shuffle(copy_x_all)
    
    
    #rs = np.random.RandomState(
    #shuffle = rs.permutation(np.range(C))

    L,F = copy_x_all.shape
    N = math.ceil(L * frac_test)
    M = L - N
    print(M)
    print(N)

    x_train_MF = np.zeros((M, F))

    x_test_NF = np.zeros((N, F))

    for i in range(N):
        x_test_NF[i] = copy_x_all.copy()[i]

    count = 0
    for a in range(N,M):
        x_train_MF[count] = copy_x_all.copy()[a]
        count= count + 1



    # TODO fixme
    return x_train_MF, x_test_NF


def calc_k_nearest_neighbors(data_NF, query_QF, K=1):
    ''' Compute and return k-nearest neighbors under Euclidean distance
    Args
    ----
    data_NF : 2D np.array, shape = (n_examples, n_features) aka (N, F)
        Each row is a feature vector for one example in dataset
    query_QF : 2D np.array, shape = (n_queries, n_features) aka (Q, F)
        Each row is a feature vector whose neighbors we want to find
    K : int, must satisfy K >= 1 and K <= n_examples aka N
        Number of neighbors to find per query vector
    Returns
    -------
    neighb_QKF : 3D np.array, (n_queries, n_neighbors, n_feats) (Q, K, F)
        Entry q,k is feature vector of k-th nearest neighbor of the q-th query
        If two vectors are equally close, then we break ties by taking the one
        appearing first in row order in the original data_NF array
    Examples
    --------
    >>> data_NF = np.asarray([
    ...     [1, 0],
    ...     [0, 1],
    ...     [-1, 0],
    ...     [0, -1]])
    >>> query_QF = np.asarray([
    ...     [0.9, 0],
    ...     [0, -0.9]])
    >>> neighb_QKF = calc_k_nearest_neighbors(data_NF, query_QF, K=1)
    >>> neighb_QKF[0]
    array([[1., 0.]])
    >>> neighb_QKF[1]
    array([[ 0., -1.]])
    # Find 3 nearest neighbors
    >>> neighb_QKF = calc_k_nearest_neighbors(data_NF, query_QF, K=3)
    >>> neighb_QKF.shape
    (2, 3, 2)
    >>> neighb_QKF[0]
    array([[ 1.,  0.],
           [ 0.,  1.],
           [ 0., -1.]])
    >>> neighb_QKF[1]
    array([[ 0., -1.],
           [ 1.,  0.],
           [-1.,  0.]])
    '''
    # TODO fixme


    #find Q, K, F
    Q = query_QF.shape[0]
    N, F = data_NF.shape[0], data_NF.shape[1]


    neighb_QKF = np.zeros((Q, K, F))
    #dtype = np.float64

    for i in range(Q):
        dist = []
        for a in range(N):
            curr_dist = np.sqrt(np.dot(query_QF[i]-data_NF[a], query_QF[i]-data_NF[a]))
            dist.append([curr_dist, data_NF[a]])
            
        dist.sort(key=takeFirst)            
        
        for b in range(K):
            neighb_QKF[i,b] = dist[b][1]
            
    return neighb_QKF


def takeFirst(elem):
    return elem[0]



