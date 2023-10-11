# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 01:02:35 2023

@author: hamid & kaddami
"""

import numpy as np
import scipy.stats as st

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
    
    return eps * X_1 + (1-eps) * X_0, eps



A = 2
N = 2
theta = A/np.sqrt(2) * np.ones((N,1))
p1 = .2
M = 100_000


data = data_generator(M, p1, theta, N)
X = data[0]
eps = data[1]
### MPE test error rate ###
MPE_test = X.T @ theta > sig**2 * np.log((1-p1)/p1) + np.sum(theta*theta)/2
err_rate_MPE = np.sum(np.abs(MPE_test.T - eps))/M
print(f"MPE test error rate : {err_rate_MPE:.4f}")


### NP error rate ###
gamma = 1e-3
NP_test = X.T @ theta > sig * np.linalg.norm(theta) * st.norm.ppf(1-gamma)
err_rate_NP = np.sum(np.abs(NP_test.T - eps))/M
print(f"NP test error rate : {err_rate_NP:.4f}")






