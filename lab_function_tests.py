# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 01:02:35 2023

@author: hamid & kaddami
"""

import numpy as np
import matplotlib.pyplot as plt
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
    covariance_matrix = sig * np.identity(N)

    # Generate random realizations
    X_0 = np.random.multivariate_normal(np.zeros(N), covariance_matrix, M)
    X_1 = np.random.multivariate_normal(theta, covariance_matrix, M)
    eps = np.random.binomial(1, p1, M)

    return eps * X_1.T+ (1-eps) * X_0.T, eps

A = 2
N = 2
theta = A/np.sqrt(2) * np.ones(N)
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


### Ananlytical expression of MPE & NPE proba of error ###
rho = np.linalg.norm(theta)/sig
Pe_MPE = (1-p1) * (1 - st.norm.cdf(1/rho * np.log((1-p1)/p1) + rho/2)) + p1 * st.norm.cdf(1/rho * np.log((1-p1)/p1) - rho/2)
Pe_NP = (1-p1) * gamma + p1 * st.norm.cdf(st.norm.ppf(1-gamma)-rho)

print(f"MPE test error probability : {Pe_MPE:.4f}") 
print(f"NP test error probability : {Pe_NP:.4f}") 


#%% plots
M = [10**x for x in range(2,7)]
err_rates_MPE = np.empty(len(M))
err_rates_NP = np.empty(len(M))

for idx, iM in enumerate(M) : 
    data = data_generator(iM, p1, theta, N)
    X = data[0]
    eps = data[1]
    
    ### MPE test error rate ###
    MPE_test = X.T @ theta > sig**2 * np.log((1-p1)/p1) + np.sum(theta*theta)/2
    err_rates_MPE[idx] = np.sum(np.abs(MPE_test.T - eps))/iM
    
    ### NP error rate ###
    NP_test = X.T @ theta > sig * np.linalg.norm(theta) * st.norm.ppf(1-gamma)
    err_rates_NP[idx] = np.sum(np.abs(NP_test.T - eps))/iM


fig = plt.figure(1, figsize=(6,6))    
ax = fig.add_subplot(111)
ax.semilogx(M, err_rates_MPE, marker='o', label = 'MPE')
ax.semilogx(M, err_rates_NP, marker='o', label = 'NP')
ax.axhline(y = Pe_MPE, color = 'r', label = r'$\mathbb{P}_e(g_{MPE})$')
ax.axhline(y = Pe_NP, color = 'b', label = r'$\mathbb{P}_e(g_{NP})$')

plt.xlabel("Monte carlo number")
plt.ylabel("Error rate")
ax.legend()
plt.title(" Error rates and probabilities of error of NP and MPE tests")
plt.savefig("Error_rate.pdf", bbox_inches='tight')
plt.show()

# Numerical verification of inequality question 4
LHS = 1 - st.norm.cdf(A/2)
RHS = 1/2 * (gamma + st.norm.cdf(st.norm.ppf(1-gamma)-A))
print(f"Left hand side of the inequality of question 4 : {LHS}") 
print(f" Right hand side of the inequality of question 4: {RHS}") 



