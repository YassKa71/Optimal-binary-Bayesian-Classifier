# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 01:02:35 2023

@author: hamid & kaddami
"""

import numpy as np


sig = 1 # std
def data_generator(M, p1, theta, N) : 
    """
    
    Parameters
    ----------
    M : int
        number of realisations.
    p1 : float between (0,1)
        prior probability for the hypothesis H_1.
    theta : numpy float array of size Nx1
        mean of data under H_1.
    N : int
        dimension of a data vector.

    Returns
    -------
    M realization of the vector X in numpy array form of size NxM.

    """
    X_0 = np.random.normal(loc = 0, scale = sig, size = (N,M))
    X_1 = X_0 + theta
    eps = np.random.binomial(1, p1, M)
    
    return eps * X_1 + (1-eps) * X_0

