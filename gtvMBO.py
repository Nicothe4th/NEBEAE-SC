#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:45:45 2024

@author: jnmc
"""

from time import time
import numpy as np
import scipy.io as sio
from sklearn.cluster import KMeans
from scipy.linalg import svd, pinv,  sqrtm, eigh
from scipy.optimize import lsq_linear
import matplotlib.pyplot as plt
import itertools

def estimate_snr(R, r_m, x):
    """
    Estimates SNR based on data, mean, and projected data.

    Args:
        R (np.ndarray): Input data matrix.
        r_m (np.ndarray): Mean of each band in the data.
        x (np.ndarray): Projected data onto the p-subspace.

    Returns:
        float: Estimated SNR value.
    """

    L, N = R.shape  # L: number of bands, N: number of pixels
    p, _ = x.shape  # p: number of endmembers (reduced dimension)

    P_y = np.sum(R**2).flatten() / N
    P_x = np.sum(x**2).flatten() / N + np.dot(r_m, r_m)
    snr_est = 10 * np.log10((P_x - p / L * P_y) / (P_y - P_x))
    return snr_est


def EIA_VCA(data, runs, p, verbosity=True):
    # Parameters
    if runs <= 0:
        raise ValueError('runs must be a positive integer')
    if p <= 0:
        raise ValueError('p must be a positive integer')
    if not isinstance(verbosity, bool):
        raise ValueError('verbosity must be a logical value')
    if verbosity:
        print('Running VCA algorithm ...')

    # Data size
    _, nsamples = data.shape

    # SNR Estimates
    if verbosity:
        print('... estimating SNR')
    r_m = np.mean(data, axis=1)
    R_m = np.tile(r_m, (nsamples, 1)).T  # mean of each band
    R_o = data - R_m  # data with zero-mean
    Ud, _, _ = svd(R_o @ R_o.T / nsamples)  # computes the p-projection matrix
    Ud = Ud[:, :p]
    x_p = Ud.T @ R_o  # project the zero-mean data onto p-subspace
    SNR = estimate_snr(data, r_m, x_p)
    SNR_th = 15 + 10 * np.log10(p)

    # Choosing Projective Projection or projection to p-1 subspace
    if SNR < SNR_th:
        if verbosity:
            print('... projective projection')
        d = p - 1
        Ud = Ud[:, :d]
        Rp = Ud @ x_p[:d, :] + np.tile(r_m, (nsamples, 1)).T  # again in dimension L
        x = x_p[:d, :]  # x_p = Ud' * R_o; is on a p-dim subspace
        c = np.sqrt(np.max(np.sum(x**2, axis=0)))
        y = np.vstack([x, c * np.ones(nsamples)])
    else:
        if verbosity:
            print('... projection to p-1 subspace')
        d = p
        Ud, _, _ = svd(data @ data.T / nsamples)  # computes the p-projection matrix
        Ud = Ud[:, :p]
        x_p = Ud.T @ data
        Rp = Ud @ x_p[:d, :]  # again in dimension L (note that x_p has no null mean)
        x = Ud.T @ data
        u = np.mean(x, axis=1)  # equivalent to u = Ud' * r_m
        y = x / np.tile(np.sum(x * np.tile(u, (nsamples, 1)).T, axis=0), (d, 1))

    # Initialization
    results = [{'E': None, 'C': None, 't': None} for _ in range(runs)]

    # Runs
    for r in range(runs):
        if verbosity:
            print(f'... run {r+1}')
        start_time = time()
        C = np.zeros(p, dtype=int)
        A = np.zeros((p, p))
        A[p-1, 0] = 1
        for i in range(p):
            w = np.random.rand(p)
            f = w - A @ pinv(A) @ w
            f = f / np.sqrt(np.sum(f**2))
            v = f @ y
            C[i] = np.argmax(np.abs(v))
            A[:, i] = y[:, C[i]]  # same as x[:, C(i)]
        E = Rp[:, C]
        elapsed_time = time() - start_time
        if verbosity:
            print(f'... finished in {elapsed_time:.2f} seconds')
        results[r]['E'] = E
        results[r]['C'] = C
        results[r]['t'] = elapsed_time

    return results

def batchvca(pixels, p, bundles, percent, seed=None):
    """
    Function performing VCA several times on randomly sampled subsets of a
    dataset, and clustering the obtained signatures into bundles using
    kmeans with the spectral angle as a similarity measure.
    
    Parameters:
    pixels : ndarray
        Data pixels, L*N
    p : int
        Number of endmember classes
    bundles : int
        Number of subsets of the data
    percent : float
        Percentage of the data taken each time
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    groups : ndarray
        Cluster labels for each signature
    components : ndarray
        Clustered components
    """
    if seed is not None:
        np.random.seed(seed)  # for reproducibility

    B = []
    m = percent / 100 * pixels.shape[1]
    runs = 1
    pixels_update = pixels.copy()

    for b in range(bundles):
        # Randomly sample without replacement
        idx = np.random.choice(pixels_update.shape[1], size=int(m), replace=False)
        C = pixels_update[:, idx]
        B.append(EIA_VCA(C, runs, p, verbosity=False)[0])
        pixels_update = np.delete(pixels_update, idx, axis=1)

    components = np.hstack([B[i]['E'] for i in range(bundles)])

    # Clustering part using kmeans with cosine distance
    kmeans = KMeans(n_clusters=p, random_state=seed, n_init=10)
    kmeans.fit(components.T)
    groups = kmeans.labels_

    return groups, components
def FCLSU(HIM, M):
    _, ns = HIM.shape
    l, p = M.shape
    Delta = 1/1000  # should be a small value
    N = np.zeros((l + 1, p))
    N[:l, :p] = Delta * M
    N[l, :] = np.ones(p)
    s = np.zeros(l + 1)

    out = np.zeros((ns, p))

    for i in range(ns):
        s[:l] = Delta * HIM[:, i]
        s[l] = 1
        result = lsq_linear(N, s, bounds=(0, np.inf))
        out[i, :] = result.x

    return out

def bundle2global(A_bundle, bundle, groups):
    """
    This function sums abundance maps corresponding to the same group and 
    computes pixelwise endmembers from the variant abundances.

    Parameters:
    - A_bundle: Q x N abundance matrix of all considered signatures
    - bundle: L x Q matrix containing these signatures
    - groups: Q x 1 vector indicating the group structure of the abundance matrix.
              Values have to range between 1 and P, the number of groups (endmembers).

    Returns:
    - A_global: P x N abundance matrix with summed abundances for each group
    - sources_global: L x P x N array of pixelwise endmembers
    """
    N = A_bundle.shape[1]
    L = bundle.shape[0]
    nbg = np.max(groups)

    A_global = np.zeros((nbg+1, N))
    sources_global = np.zeros((L, nbg+1, N))

    threshold = 10 ** (-2)

    # Replace values below the threshold with 0
    A_bundle_new = A_bundle.copy()
    A_bundle_new[np.abs(A_bundle) < threshold] = 0

    for p in range(0, nbg+1):
        A_global[p, :] = np.sum(A_bundle_new[groups == p, :], axis=0)
        
        for i in range(N):
            if A_global[p, i] != 0:
                sources_global[:, p, i] = np.sum((np.tile(A_bundle_new[groups == p, i],(L,1)) * bundle[:, groups == p]), axis=1) / A_global[p, i]
            else:
                sources_global[:, p, i] = np.mean(bundle[:, groups == p], axis=1)
    
    return A_global, sources_global

def find_perm(A_true, A_hat, S_hat):
    """
    Find the permutation of A_hat and S_hat that minimizes nMSE compared to A_true.
    
    Parameters:
    A_true : ndarray
        True endmember matrix [P x L]
    A_hat : ndarray
        Estimated endmember matrix [P x L]
    S_hat : ndarray
        Estimated abundance matrix [L x N]
    
    Returns:
    A_hat : ndarray
        Permuted estimated endmember matrix [P x L]
    S_hat : ndarray
        Permuted estimated abundance matrix [L x N]
    nmse : float
        Normalized mean squared error
    """
    P = A_true.shape[0]
    ords = np.array(list(itertools.permutations(range(P))))  # All permutations of indices 1 to P
    n = ords.shape[0]
    errs = np.ones(n) * 100  # Initialize errors with a large value
    
    for idx in range(n):
        perm_indices = ords[idx, :]
        A_permuted = A_hat[perm_indices, :]
        errs[idx] = nMSE(A_true, A_permuted)
    
    I = np.argmin(errs)
    nmse = errs[I]
    
    # Apply the best permutation found
    best_perm_indices = ords[I, :]
    A_hat = A_hat[best_perm_indices, :]
    S_hat = S_hat[:, best_perm_indices]
    
    return A_hat, S_hat, nmse

def nMSE(x, xhat):
  return np.linalg.norm(x - xhat, 'fro') / np.linalg.norm(x, 'fro')



def laplacian_nystrom(Xt, metric, num_samples, sigma2, seed=None):
    """
    Laplacian Nystrom method for spectral clustering.

    Parameters:
    Xt : ndarray
        Matrix transpose (#pixels x #wavelengths)
    metric : int
        Distance metric to use (1 for Euclidean squared distance, 
        2 for acos similarity, 3 for cos similarity)
    num_samples : int
        Number of eigenvalues/vectors, also the number of random samples
    sigma2 : float
        Variance constant used in computing similarity
    seed : int, optional
        Seed for random number generator for reproducibility

    Returns:
    V : ndarray
        Eigenvectors of size #pixels x #samples
    L : ndarray
        Eigenvalues of Laplacian #samples x #samples
    """
    if seed is not None:
        np.random.seed(seed)

    # Randomly select samples
    num_rows = Xt.shape[0]
    permed_index = np.random.permutation(num_rows)
    sample_data = Xt[permed_index[:num_samples], :]
    other_data = Xt[permed_index[num_samples:], :]
    
    # Calculate the distance between samples themselves
    A = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        for j in range(i, num_samples):
            x = sample_data[i, :]
            y = sample_data[j, :]

            if metric == 1:
                d = np.sum((x - y) ** 2)
            elif metric == 2:
                d = np.arccos(np.clip(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)), -1.0, 1.0))
            elif metric == 3:
                d = 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

            A[i, j] = np.exp(-d / sigma2)
            A[j, i] = A[i, j]

    # Calculate the distance between samples and other points
    other_points = num_rows - num_samples
    B = np.zeros((num_samples, other_points))
    for i in range(num_samples):
        for j in range(other_points):
            x = sample_data[i, :]
            y = other_data[j, :]

            if metric == 1:
                d = np.sum((x - y) ** 2)
            elif metric == 2:
                d = np.arccos(np.clip(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)), -1.0, 1.0))
            elif metric == 3:
                d = 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

            B[i, j] = np.exp(-d / sigma2)

    # Normalize A and B using row sums of W
    B_T = B.T
    d1 = np.sum(A, axis=1) + np.sum(B, axis=1)
    d2 = np.sum(B_T, axis=1) + B_T @ (pinv(A) @ np.sum(B, axis=1))
    dhat = np.sqrt(1.0 / np.concatenate((d1, d2)))

    A = A * (dhat[:num_samples][:, None] @ dhat[:num_samples][None, :])
    B = B * (dhat[:num_samples][:, None] @ dhat[num_samples:][:, None].T)

    # Orthogonalization and eigendecomposition
    Asi = sqrtm(pinv(A))
    B_T = B.T
    BBT = B @ B_T
    W = np.vstack([A, B_T])
    
    R = A + Asi @ BBT @ Asi
    R = (R + R.T) / 2  # Symmetrize R
    L, U = eigh(R)
    ind = np.argsort(L)[::-1]
    U = U[:, ind]  # In decreasing order
    L = np.diag(L[ind])  # In decreasing order

    W = W @ Asi
    V = W @ U[:, :num_samples] @ pinv(np.sqrt(L[:num_samples, :num_samples]))

    # Permute the eigenvectors back to the original order
    V_temp = np.zeros_like(V)
    V_temp[permed_index, :] = V
    V = np.real(V_temp)
    L = 1 - np.diag(L)

    return V, L

def SimplexProj2(y):
    """
    Projects the columns of y onto the probability simplex.
    
    Parameters:
    y : ndarray
        Input matrix where each column will be projected onto the simplex
    
    Returns:
    A : ndarray
        Output matrix with the same shape as y, where each column is the projection of the corresponding column in y onto the simplex
    """
    # Sort y in descending order along each column
    sorted_y = np.sort(y, axis=0)[::-1]
    
    # Compute the cumulative sum of sorted_y along each column
    cumsum_y = np.cumsum(sorted_y, axis=0)
    
    # Create an index array for the divisor
    index = np.arange(1, y.shape[0] + 1).reshape(-1, 1)
    
    # Compute the maximum value for each column
    tmp = (cumsum_y - 1) / index
    max_tmp = np.max(tmp, axis=0)
    
    # Subtract max_tmp from y and apply the max function with 0
    A = np.maximum(y - max_tmp, 0)
    
    return A

def gtvMBO(V, Lambda, Y, mu, tol, dt):
    # Inputs:
    # V = eigenvectors of approximate Ls
    # Lambda = eigenvalues of approximate Ls
    # Y: a matrix of P x N (N-number of points, P-number of features)
    # Outputs:
    # sol: solution to min_u |u|_{gTV}+|u_t|+mu|u-y|^2

    # Setup other input parameters
    Y = Y.T  # size N x P
    N, P = Y.shape
    K = len(Lambda)  # number of singular values
    U0 = np.zeros((N, P))

    maxiters = 5
    num_bits = 8
    denom = 1 - dt * Lambda

    # Decompose Y into 8 channels
    Ychannel = np.zeros((N, P, num_bits), dtype=np.uint8)
    for a in range(P):
        t = np.clip(np.ceil(Y[:, a] * 255), 0, 255).astype(np.uint8)
        M1 = np.unpackbits(t[:, None], axis=1, bitorder='little')[:, :num_bits]
        Ychannel[:, a, :] = M1

    # Run iters for each channel
    Uchannels = np.zeros((N, P, num_bits), dtype=np.uint8)
    for e in range(num_bits):
        U = U0  # Initialize U for each channel

        # Set up iterations for each channel
        converged = False
        iter = 1

        a = V.T @ U  # K x P
        d = np.zeros((K, P))
        Yslice = Ychannel[:, :, e]

        while iter <= maxiters and not converged:
            # Get U update
            a = np.diag(denom) @ a - dt * d
            U = V @ a
            d = mu * V.T @ (U - Yslice)  # K x P

            # Threshold
            Unext = np.zeros_like(U)
            Unext[U >= 0.5] = 1

            # Check relative diff in update
            iter += 1
            err = np.linalg.norm(U - Unext, 'fro') / np.linalg.norm(U, 'fro')
            if err <= tol:
                converged = True
            U = Unext

        Uchannels[:, :, e] = U

    # Recombine Ui
    Uout = np.zeros((N, P), dtype=np.uint8)
    for h in range(P):
        Uout[:, h] = np.packbits(Uchannels[:, h, :], axis=1, bitorder='little').flatten()

    sol = (Uout / 255).T
    sol = np.clip(sol, 0, 1)

    return sol


def unmixing(X, S0, A0, para):
    K = S0.shape[1]  # Number of endmembers
    # Initialization
    S = S0.copy()
    A = A0.copy()
    B = A.copy()
    Btilde = np.zeros_like(A)
    Ctilde = np.zeros_like(S)
    for i in range(para["itermax"]):
        if (i + 1) % 50 == 0 and para["print_flag"]:
        # Print iteration number, condition number of matrices (optional)
            print(f"iter {i + 1}")
            # You can uncomment these lines to check for ill-conditioning
            # print(np.linalg.cond(A.T @ A))
            # print(np.linalg.cond(S.T @ S))

        Aprev = A.copy()
        Sprev = S.copy()
        
        # C-subproblem
        C = (X @ A.T + para["gamma"] * (S + Ctilde)) @ np.linalg.inv( (A @ A.T + para["gamma"] * np.eye(K)))
        
        # S-subproblem
        S = np.maximum(0, C - Ctilde)
        
        # A-subproblem
        LHS = S.T @ S + para["rho"] * np.eye(K)
        RHS = S.T @ X + para["rho"] * (B - Btilde)
        A = np.linalg.solve(LHS, RHS)
        
        if np.isnan(A).any() or np.isnan(S).any():
            return S, A, i
    
        if np.max(A) > 10**3 or np.max(S) > 10**3:
            return S, A, i
        
        # Projection onto the probability simplex
        A = SimplexProj2(A)
        
        # B-subproblem (gtvMBO method)
        tmp = A + Btilde
        mu = para["rho"] / para["lambda"]
        B = gtvMBO(para["V"], para["S"], tmp, mu, para["tol"], para["dt"])
        
        # Update auxiliary variables
        Btilde += A - B
        Ctilde += S - C
        
        # Stopping criteria
        if np.linalg.norm(Sprev - S) / np.linalg.norm(Sprev) < para["tol"] and np.linalg.norm(Aprev - A) / np.linalg.norm(Aprev) < para["tol"]:
            break
    
    return S, A, i

def gtvMBO_unmix(X,S_init,P,tol,m,n,sigma):
    X = X/np.max(X)
    _, N = X.shape;
    seed =1;
    np.random.seed(seed);
    # bundle_nbr = 10; # number of VCA runs
    # percent = 10; # percentage of pixels considered in each run
    # print('vcabatcha')
    # groups, bundle = batchvca(X, P, bundle_nbr, percent, seed)
    # S_init = [];
    
    # print('idxP')
    # for idx in range(P):
    #     s = np.mean(bundle[:,groups == idx], axis=1)
    #     S_init.append(s)
    S_init = np.maximum(S_init, 0)
    
    bundle = S_init.copy()
    
    A_init =  FCLSU(X, bundle).T;
    #A_FCLSU = FCLSU(X, bundle).T;
    #A_init, _ = bundle2global(A_FCLSU,bundle,groups);
    
    #A_init, S_init, nmse = find_perm(A_ref, A_init, S_init);


    V, Sigma = laplacian_nystrom(X.T, 3, int(np.floor(0.001*N)), sigma, seed);
    
    
    
    para_mbo = {
        'tol': tol,
        'm': m,
        'n': n,
        'V': V,
        'S': Sigma,
        'itermax': 100,
        'dt': 0.1,
        'lambda': 10**(-4.25),
        'rho': 10**(-2.75),
        'gamma': 10**(3.75),
        'print_flag':False
    }
    start_time = time()
    S_MBO, A_MBO, iter = unmixing(X, S_init, A_init, para_mbo)
    t_MBO = time() - start_time
    #A_MBO, S_MBO, _ = find_perm(A_ref, A_MBO, S_MBO);
    S_MBO = S_MBO/np.sum(S_MBO,axis=0,keepdims=True)
    A_MBO = A_MBO / np.sum(A_MBO, axis=0, keepdims=True)  
    Y_MBO = S_MBO@A_MBO
    return A_MBO, S_MBO, Y_MBO, t_MBO






