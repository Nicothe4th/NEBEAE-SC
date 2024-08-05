#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 10:55:38 2023

@author: jnmc
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

def vca(Y,R):
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
    Ud  = np.linalg.svd(np.dot(Y_o,Y_o.T)/float(N))[0][:,:R]  # computes the R-projection matrix 
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
        Ud  = np.linalg.svd(np.dot(Y,Y.T)/float(N))[0][:,:d] # computes the p-projection matrix 
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
        f = w - np.dot(A,np.dot(np.linalg.pinv(A),w))
        f = f / np.linalg.norm(f)
        v = np.dot(f.T,y)
        indice[i] = np.argmax(np.abs(v))
        A[:,i] = y[:,indice[i]]        # same as x(:,indice(i))
    Ae = Yp[:,indice]
    return Ae

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

def fl(P,A,D,X):
    L,p = P.shape
    s,N =A.shape
    y= P @ A
    l=np.zeros(N)
    
    for i in range(N):
        xi = X[:,i];
        yi = y[:,i];
        di = D[i,0].flatten()
        ti = xi - ( (1-di)*yi ) - ( di * yi * xi)
        l[i] = np.linalg.norm(ti,ord=2)**2
    return  l.sum()


def upD(X, P, A):
    L, N = X.shape
    s, p = P.shape
    Y = P @ A # Y = Ea
    D_g = np.zeros((N, 1))
    for i in range(N):
        t1 = Y[:, i] - (Y[:, i] * X[:, i])
        den = np.linalg.norm(t1,ord=2) ** 2 # ||y - y.*x||_2^2
        num = t1.T @ (Y[:, i] - X[:, i])
        D_g[i] = min(1, num / den) # Eq. (14)
    return D_g

def projsplx(y):
    m = len(y)
    bget = False
    s = np.sort(y)[::-1]
    tmpsum = 0
    for ii in range(m - 1):
        tmpsum = tmpsum + s[ii]
        tmax = (tmpsum - 1) / (ii + 1)
        if tmax >= s[ii + 1]:
            bget = True
            break
    if not bget:
        tmax = (tmpsum + s[m - 1] - 1) / m
    x = np.maximum(y - tmax, 0)
    return x

def mina(Y, Po, Ao, Do):
    L, N = Y.shape
    s, p = Po.shape
    Itm = np.ones((1, p))
    Idm = np.ones((L, p))
    a_g = np.zeros((p, N))
    for i in range(N):
        P_g = (Po * (1 -Do[i]) * Idm) + (Do[i] * Y[:,i] @ Idm)
        L = np.linalg.norm(P_g.T @ P_g, 'fro')
        g_ga = P_g.T @ ( P_g@Ao[:,i] - Y[:,i])
        a_g[:, i] = projsplx(Ao[:, i] - g_ga / L)
    return a_g

def minp(P, A, X, D):
    # optimization respect E
    # E endmember matrix
    # A abundance matrix
    # X Hyperspectral image
    # P probability
    p, N = A.shape
    L, p = P.shape
    A_i = np.zeros((L, p, N))
    gPp = np.zeros((L, p, N))

    for i in range(N):
        A_i[:, :, i] = ( (1 - D[i]) * np.ones((L, 1)) + (D[i]* X[:, i]).reshape(-1,1)  ) @ A[:,i].reshape(-1,1).T
        gPp[:, :, i] = ( ((P * A_i[:,:,i]) @ np.ones((p,1)) -  X[:,i].reshape(-1,1)) @ np.ones((1,p)) ) * A_i[:,:,i]

    gP=np.sum(gPp,axis=2);
    gp=np.zeros((p,p));
    sn=np.zeros((p,p,N)); 
    P_g = np.zeros((L, p))


    for j in range (L):
        for i in range(N):
            sn[:,:,i]=(A_i[j,:,i] * A_i[j,:,i].T);
        gp[:,:]=np.sum(sn,axis=2);
        fnorm=np.linalg.norm(gp);
        P_g[j,:]= P[j,:] - ( gP[j,:]/fnorm );
    P_g = np.minimum(P_g, np.ones((L, p)))
    P_g = np.maximum(P_g, np.zeros((L, p)))
    return P_g


    

def UNSUBMM(Y=[], n=[], Po=[], itmax = 20):
    [L, N]=Y.shape
    if (type(Po) != np.ndarray):
        print('vcaaaa')
        Po = vca (Y,n)
    #Do = np.zeros((N,1))
    Ao = LMM(Y, Po);
    
    P_t = Po.copy()
    A_t = Ao.copy()
    D_t1 = upD(Y, Po, Ao)
    D_t =D_t1.copy()
    e = 1e-3
    l = np.zeros((itmax+1,))
    l[0] = fl(P_t,A_t,D_t1,Y)
    
    for i in range(1, itmax):
        A_t1 = mina(Y, P_t, A_t, D_t)
        if (fl(P_t,A_t1,D_t,Y) < fl(P_t,A_t,D_t,Y)):
            A_t = A_t1.copy()
        D_t = upD(Y, P_t, A_t)
        
        P_t1 = minp(P_t, A_t, Y, D_t) # update E algo2
        if (fl(P_t1, A_t, D_t, Y) <= fl(P_t, A_t, D_t, Y)):
            P_t = P_t1.copy()
            
        l[i] = fl(P_t, A_t, D_t, Y)
        if(abs((l[i] - l[i-1]) / l[i-1]) < e):
            break
    ylm = P_t @ A_t
    X = np.zeros((L, N))
    D_t = upD(Y, P_t, A_t)
    for j in range(N):
        X[:, j] = ((1 - D_t[j]) * ylm[:, j]) / (1 - (D_t[j] * ylm[:, j]))
    return P_t, A_t, D_t, X, l