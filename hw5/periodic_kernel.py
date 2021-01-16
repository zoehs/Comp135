'''
Defines function `calc_periodic_kernel`

Examples
--------
>>> np.set_printoptions(precision=3, suppress=1)

# Part 1: Simple kernel evaluations with F=1 features
>>> x_zero_11 = np.asarray([[0.0]])
>>> x_one_11 = np.asarray([[1.0]])

# Kernel of x=0.0 with itself should be 1.0
>>> k_11 = calc_periodic_kernel(x_zero_11, x_zero_11, length_scale=2.0, period=0.3)
>>> k_11.ndim
2
>>> k_11
array([[1.]])

# Kernel of x=1.0 with itself should be 1.0
>>> calc_periodic_kernel(x_one_11, x_one_11, length_scale=2.0, period=0.3)
array([[1.]])

# Kernel of x and z=x+period should be 1.0
>>> p = 0.3
>>> calc_periodic_kernel(x_one_11, x_one_11 + p, length_scale=2.0, period=p)
array([[1.]])
>>> calc_periodic_kernel(x_one_11, x_one_11 + 3 * p, length_scale=2.0, period=p)
array([[1.]])

# A few other tests
>>> calc_periodic_kernel(x_one_11, x_zero_11, length_scale=0.5, period=0.3)
array([[0.223]])
>>> calc_periodic_kernel(x_one_11, x_zero_11, length_scale=2.0, period=0.95)
array([[0.997]])

# Part 2: Kernel evaluations with several examples at once (still F=1)
>>> x_train_31 = np.asarray([[0.0], [1.0], [2.0]])
>>> calc_periodic_kernel(x_train_31, x_train_31, length_scale=2.0, period=0.95)
array([[1.   , 0.997, 0.987],
       [0.997, 1.   , 0.997],
       [0.987, 0.997, 1.   ]])
>>> x_test_21 = np.asarray([[-0.5], [0.5]])
>>> calc_periodic_kernel(x_test_21, x_train_31, length_scale=2.0, period=0.95)
array([[0.883, 0.889, 0.9  ],
       [0.883, 0.883, 0.889]])
'''

import numpy as np

def calc_periodic_kernel(x_QF, x_train_NF=None, length_scale=1.0, period=1.0):
    ''' Evaluate periodic kernel to produce matrix between two datasets.

    Will compute the kernel function for all possible pairs of feature vectors,
    one from the query dataset, one from the reference training dataset.

    Args
    ----
    x_QF : 2D numpy array, shape (Q, F) = (n_query_examples, n_features)
        Feature array for *query* dataset
        Each row corresponds to the feature vector on example

    x_train_NF : 2D numpy array, shape (N, F) = (n_train_examples, n_features)
        Feature array for reference *training* dataset
        Each row corresponds to the feature vector on example
        
    Returns
    -------
    k_QN : 2D numpy array, shape (Q, N)
        Entry at index (q,n) corresponds to the kernel function evaluated
        at the feature vectors x_QF[q] and x_train_NF[n]
    '''
    assert x_QF.ndim == 2
    assert x_train_NF.ndim == 2

    Q, F = x_QF.shape
    N, F2 = x_train_NF.shape
    assert F == F2
    
    k_QN = np.zeros((Q, N))
    # TODO compute kernel between rows of x_QF and rows of x_train_NF
    for i in range(Q):
        for a in range(N):
            up = np.sin(np.pi * ((x_QF[i]-x_train_NF[a]))/period)**2
            down = 2*(length_scale**2)
            k_QN[i,a] = np.exp(up/-down)




    # Ensure the kernel matrix positive definite
    # By adding a small positive to the diagonal
    M = np.minimum(Q, N)
    k_QN[:M, :M] += 1e-08 * np.eye(M)
    return k_QN





