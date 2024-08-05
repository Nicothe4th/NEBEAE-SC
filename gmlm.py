
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 14:02:30 2024

@author: nicolasmendoza
"""
import numpy as np
from skimage.segmentation import slic
import scipy.linalg as splin

def LMM(Y, Po):
    num_endmembers = Po.shape[1]
    num_pixels = Y.shape[1]
    A = np.zeros((num_endmembers, num_pixels))
    max_iter = 5
    tol = 1e-3
    for i in range(max_iter):
    # Update the abundances matrix
        A_old = A.copy()
        A = np.linalg.lstsq(Po, Y, rcond=None)[0]
        A[A < 0] = 0
        A = A / np.sum(A, axis=0, keepdims=True)
    # Compute the objective function

    # Check for convergence 
        if np.linalg.norm(A - A_old) < tol:
            #print("Converged after", i+1, "iterations.")
            break
    return A


def Ec29(Prt, rho, M2, l3, W):
    N, _ = W.shape
    sumw = np.sum(W, axis=1)
    L = np.diag(sumw) - W
    H = rho * (Prt + M2) @ np.linalg.inv(l3 * L + rho * np.ones((N, N)))
    Ht = np.minimum(1, H)
    return Ht

def Ec26(Po, At, X, rho, Ht, M2):
    L, N = X.shape
    n, m = At.shape
    Y = Po @ At
    numerator = np.sum((Y - Y * X) * (Y - X), axis=0) + rho * (Ht - M2)
    denominator = np.sum((Y - Y * X) ** 2, axis=0) + rho * np.ones(N)
    Prt = numerator / denominator
    return Prt

def Ec24(At, M1, l1, rho):
    b = l1 / rho
    Z = At + M1
    C1r, C1c = np.where(Z > b)
    C3r, C3c = np.where(np.abs(Z) < -b)
    S = np.zeros_like(At)

    if C1r.size != 0:
        S[C1r, C1c] = Z[C1r, C1c] - b

    if C3r.size != 0:
        S[C3r, C3c] = Z[C3r, C3c] + b

    Gt1 = np.maximum(0, S)
    return Gt1

def Ec17(At, Po, X, rho, Prt, l3, W, Gt, M1):
    L, N = X.shape
    n, m = At.shape
    At1 = np.zeros_like(At)

    for j in range(N):
        Pj_hat = Po * ((1 - Prt[j]) * np.ones((L, n)) + Prt[j] * X[:, j][:, None])
        B = Pj_hat.T @ Pj_hat + (rho + l3 * np.sum(W[:, j])) * np.eye(n)
        C = np.linalg.inv(B) @ np.ones((n, 1)) @ np.linalg.inv(np.ones((1, n)) @ np.linalg.inv(B) @ np.ones((n, 1)))
        #w1 = Pj_hat.T @ X[:, j]
        #print('size w1 = ', w1.shape)
        #w2 = rho * (Gt[:, j] - M1[:, j])
        #print('size w2 = ', w2.shape)
        #w3 = l3 * (np.sum( W[:, j]) *  At[:, j])
        #print('size w3 = ', w3.shape)
        w =  Pj_hat.T @ X[:, j] + rho * (Gt[:, j] - M1[:, j]) + l3 * (np.sum( W[:, j]) *  At[:, j])#w1 +  w2+ w3
        At1[:, j] = np.linalg.inv(B) @ w - C @ (np.ones((1, n)) @ np.linalg.inv(B) @ w - 1)

    return At1

def VCA(Y,R):
    #############################################
    # Initializations
    #############################################
    [L, N]=Y.shape   # L number of bands (channels), N number of pixels     
    R = int(R)
    #############################################
    # SNR Estimates
    #############################################  
    y_m = np.mean(Y,axis=1,keepdims=True)
    Y_o = Y - y_m           # data with zero-mean
    Ud  = splin.svd(np.dot(Y_o,Y_o.T)/float(N))[0][:,:R]  # computes the R-projection matrix 
    x_p = np.dot(Ud.T, Y_o)                 # project the zero-mean data onto p-subspace
    P_y     = np.sum(Y**2)/float(N)
    P_x     = np.sum(x_p**2)/float(N) + np.sum(y_m**2)
    SNR = 10*np.log10( (P_x - R/L*P_y)/(P_y - P_x) ) 
    SNR_th = 15 + 10*np.log10(R)+8
    
    #############################################
    # Choosing Projective Projection or 
    #          projection to p-1 subspace
    #############################################
    if SNR < SNR_th:
        d = R-1
        Ud = Ud[:,:d]
        Yp =  np.dot(Ud,x_p[:d,:]) + y_m      # again in dimension L
        x = x_p[:d,:] #  x_p =  Ud.T * Y_o is on a R-dim subspace
        c = np.argmax(np.sum(x**2,axis=0))**0.5
        y = np.vstack(( x, c*np.ones((1,N))))
    else:
        d=R
        Ud  = splin.svd(np.dot(Y,Y.T)/float(N))[0][:,:d] # computes the p-projection matrix 
        x_p = np.dot(Ud.T,Y)
        Yp =  np.dot(Ud,x_p[:d,:])      # again in dimension L (note that x_p has no null mean)
        x =  np.dot(Ud.T,Y)
        u =  np.mean(x,axis=1,keepdims=True)        #equivalent to  u = Ud.T * r_m
        y =  x / np.dot(u.T,x)
    #############################################
    # VCA algorithm
    #############################################
            
    indice = np.zeros((R),dtype=int)
    A = np.zeros((R,R))
    A[-1,0] = 1

    for i in range(R):
        w = np.random.rand(R,1);   
        f = w - np.dot(A,np.dot(splin.pinv(A),w))
        f = f / splin.norm(f)
        v = np.dot(f.T,y)
        indice[i] = np.argmax(np.abs(v))
        A[:,i] = y[:,indice[i]]        # same as x(:,indice(i))
    Ae = Yp[:,indice]
    return Ae,indice,Yp

def g_mlm(X, n, rho, l1, l2, l3, maxiter, Po):
    L, N = X.shape

    At = LMM(X, Po)
    elim = 1e-4

    # Initial conditions
    Y = Po @ At
    dmin = 400 * (np.linalg.norm(X - Y, 'fro') ** 2) / (N * L)

    # Affinity matrix
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            value = np.linalg.norm(X[:, i] - X[:, j]) ** 2
            if value < dmin:
                W[i, j] = 1
                W[j, i] = 1
    numerator = (np.ones((1, L)) @ (((Y - Y * X) * (Y - X))) ).flatten()
    denominator = (((np.ones((1, L)) @ ((Y - Y * X) ** 2)) + rho * np.ones((1, N)))).flatten()
    Prt = (numerator  / denominator)
    
    Gt = At.copy()
    Go = Gt.copy()
    Ht = Prt.copy()
    Ho = Ht.copy()
    M1 = np.zeros_like(At)
    M2 = np.zeros_like(Prt)

    for i in range(maxiter):
        At = Ec17(At, Po, X, rho, Prt, l3, W, Gt, M1)  # update A Ec(17)
        Gt = Ec24(At, M1, l1, rho)  # update G Ec(24)
        Prt = Ec26(Po, At, X, rho, Ht, M2)  # update Prt EC(26)
        Ht = Ec29(Prt, rho, M2, l3, W)  # update Ht Ec(29)
        M1 = M1 + At - Gt  # update M1 Ec(30)
        M2 = M2 + Prt - Ht  # update M2 Ec(31)

        resp = np.linalg.norm(np.vstack([At - Gt, Prt - Ht]), 'fro')
        resd = np.linalg.norm(np.vstack([Go - Gt, Ho - Ht]), 'fro')

        Go = Gt
        Ho = Ht

        if resp <= elim and resd <= elim:
            break

    ylm = Po @ At
    Y = np.zeros_like(X)

    for i in range(N):
        Y[:, i] = ((1 - Prt[i]) * ylm[:, i]) / (1 - (Prt[i] * ylm[:, i]))

    return Y, At, Po, Prt

def spxls_gmlm(Z, z, n, Po, nsp, maxiter, rho = 1e-5, l1 = 0.001 , l2= 0.001, l3= 0.001):
    # Superpixels
    eti = slic(z, n_segments=nsp, channel_axis=None,compactness=10)
    numeti = np.max(eti)
#1e-5, 1e-3,1e-3, 1e-3

    [L, N] = Z.shape
    #Po, _ ,_= VCA(Z / np.sum(Z, axis=0), n)

    etir = eti.flatten()
    Y = np.zeros((L, N))
    A = np.zeros((n, N))
    Prt = np.zeros(N)

    for h in range(1, numeti+1):
        n1 = np.where(etir == h)[0]
        Y[:, n1], A[:, n1], _, Prt[n1] = g_mlm(Z[:, n1], n, rho, l1, l2, l3, maxiter, Po)

    return Y, A, Po, Prt
