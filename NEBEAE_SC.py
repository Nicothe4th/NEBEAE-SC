#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 11:20:32 2024

@author: jnmc
"""

import time
import numpy as np
from scipy.sparse.linalg import svds
import scipy.linalg as splin
from scipy.sparse import spdiags
import scipy.sparse as ssp
from scipy.linalg import  pinv, eigh

def pca(X,d):
    L, N = X.shape
    xMean = X.mean(axis=1).reshape((L,1))
    xzm = xMean - np.tile(xMean, (1, N))
    U, _ , _  = svds( (xzm @ xzm.T)/N , k=d)
    return U

def NFINDR(Y, N):
    """
    N-FINDR endmembers estimation in multi/hyperspectral dataset
    """
    L, K = Y.shape
    # dimention redution by PCA
    U = pca(Y,N)
    Yr = U.T @ Y
    # Initialization
    Po = np.zeros((L,N))
    IDX = np.zeros((1,K))
    TestM = np.zeros((N,N))
    TestM[0,:]=1
    for i in range (N):
        idx = np.floor(float(np.random.rand(1))*K) + 1
        TestM[1:N,i]= Yr[:N-1,i].copy()
        IDX[0,i]=idx
    actualVolume = np.abs(np.linalg.det(TestM))
    it=1
    v1=-1
    v2=actualVolume
    #  Algorithm
    maxit = 3 * N
    while (it<maxit and v2>v1):
        for k in range (N):
            for i in range (K):
                actualSample = TestM[1:N,k].copy()
                TestM[1:N,k] = Yr[:N-1,i].copy()
                volume = np.abs(np.linalg.det(TestM))
                if volume > actualVolume:
                    actualVolume = volume.copy()
                    IDX[0,k] = i
                else:
                    TestM[1:N,k]=actualSample.copy()
        it = it + 1
        v1 = v2
        v2 = actualVolume.copy()
    
    for i in range (N):
        Po[:,i] = Y[:,int(IDX[0,i])].copy()
    return Po

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
        Ud = Ud[:,:d].copy()
        Yp =  np.dot(Ud,x_p[:d,:]) + y_m      # again in dimension L
        x = x_p[:d,:].copy() #  x_p =  Ud.T * Y_o is on a R-dim subspace
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
        A[:,i] = y[:,indice[i]].copy()        # same as x(:,indice(i))
    Ae = Yp[:,indice].copy()
    Ae[Ae < 0] = 0
    return Ae,indice,Yp

def SVMAX(X, N):
    """
    An implementation of SVMAX algorithm for endmember estimation.
    
    Parameters:
    X : numpy.ndarray
        Data matrix where each column represents a pixel and each row represents a spectral band.
    N : int
        Number of endmembers to estimate.
        
    Returns:
    A_est : numpy.ndarray
        Estimated endmember signatures (or mixing matrix).
    """
    M, L = X.shape
    d = np.mean(X, axis=1, keepdims=True)
    U = X - np.outer(d, np.ones((1, L)))
    C = eigh(U @ U.T, lower=False, subset_by_index=(M-N+1, M-1))[1]
    Xd_t = C.T @ U;
    
    A_set = np.zeros((N,N))
    
    index = np.zeros(3); 
    P = np.eye(N);    
    
                         
    Xd = np.vstack((Xd_t, np.ones((1, L))))
    
    
    for i in range(N):
        ind = np.argmax((np.sum((abs(P@Xd))**2,axis=0,keepdims=True))**(1/2));
        A_set[:,i] = Xd[:, ind]
        P = np.eye(N) - A_set @ pinv(A_set);
        index[i] = ind;
    Po = C @ Xd_t[:,index.astype(int)]+d @np.ones((1,N));
    
    
    return Po

def performance(fn):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        #print(f'Function {fn.__name__} took {t2-t1} s')
        return t2 - t1, result
    return wrapper

def abundance(Z,Y,P,W,D,Lambda, NUMERROR):
    L,K = Y.shape
    _,N= P.shape
    A = np.zeros((N,K))
    em = np.eye(N)
    
    if np.size(P,0) != L:
        print('ERROR: the number of rows in Y and P does not match');
        NUMERROR=1;
        return A, NUMERROR
    
    for k in range(K):
        yk = np.c_[Y[:,k]]
        zk = np.c_[Z[:,k]]
        wk =np.c_[W[:,k]]
        byk = np.sum(yk*yk)
        dk = D[k].copy()
        deltakn = ((1-dk)*np.ones((N,1)))+(dk*P.T@zk)
        Pk = P * ( (1-dk) * np.ones((L,N)) + (dk * zk @ np.ones((1,N))) )
        Go = Pk.T @ Pk
        eGo, _ = np.linalg.eig(Go)
        eGo[np.isnan(eGo)]=1e6
        eGo[np.isinf(eGo)]=1e6
        lmin=np.amin(eGo)
        G = Go+em*lmin*Lambda
        Gi = np.linalg.pinv(G)
        bk = Pk.T@yk + (lmin*Lambda*wk)
        ##Compute the optimal unconstrained Solution
        sigma = float((deltakn.T @Gi@bk -1) / (deltakn.T@Gi@deltakn))
        ak = Gi@(bk-deltakn*sigma)
        #A[:,k] = np.c_[ak].T
        # Check for Negative Values
        if float(sum(ak >= 0)) != N:
            I_set = np.zeros((1, N))
            while float(sum(ak < 0)) != 0:
                I_set = np.where(ak < 0, 1, I_set.T).reshape((1, N),order='F')
                L1 = len(np.where(I_set == 1)[1])
                Q = N+1+L1
                Gamma = np.zeros((Q, Q))
                Beta = np.zeros((Q, 1))
                Gamma[:N, :N] = G.copy()
                Gamma[:N, N] = (deltakn*byk).reshape((N,),order='F').copy()
                Gamma[N, :N] = deltakn.copy().T
                cont = 0
                for i in range(N):
                    if I_set[:,i] != 0:
                        cont += 1
                        ind = i
                        Gamma[ind, N+cont] = 1
                        Gamma[N+cont, ind] = 1
                Beta[:N, :] = bk.copy()
                Beta[N, :] = 1
                delta = np.linalg.solve(Gamma, Beta)
                ak = delta[:N].copy()
                ak = np.where(abs(ak) < 1e-9, 0, ak)
                ak = ak/np.sum(ak)
        A[:,k] = np.c_[ak].T
    return A, NUMERROR

def probanonlinear(Z,Y,P,A):
    L,K = Y.shape
    D=np.zeros((K,))
    for k in range(K):
        yk=np.c_[Y[:,k]]
        zk=np.c_[Z[:,k]]
        ak=np.c_[A[:,k]]
        ek=P@ak
        T1 = ek-yk
        T2 = ek-(ek*zk)
        T = float(T1.T@T2/(np.sum(T2*T2)))
        if np.isnan(T):
            T = D[k-1]
        dk = np.minimum(1, T)
        D[k]=dk.copy()
    return D

def endmember(Z,Y,Po,A,D,rho,NUMERROR):
    N,K = A.shape
    L,_ = Y.shape
    R = sum(N-np.array(range(1, N)))
    em=np.eye(N)
    onesL = np.ones((L,1))
    GradP = np.zeros((L,N))
    for k in range(K):
        yk=np.c_[Y[:,k]]
        zk=np.c_[Z[:,k]]
        ak=np.c_[A[:,k]]
        byk=yk.T@yk
        dk=D[k].copy()
        Mk= np.diag( np.squeeze((1-dk)*onesL + dk*zk))
        GradP= GradP - (Mk.T @ yk @ ak.T / byk) + ((Mk.T @Mk @ Po@ ak @ ak.T) / byk )
    O = N*em -np.ones(N)
    GradP = GradP / K + rho * Po @ (O/R)
    
    ## Compute Optimal scale Update Rule
    
    numG = rho * np.trace(GradP @ O @ Po.T + Po @ O @ GradP.T)/R/2
    denG = rho * np.trace(GradP @ O @ GradP.T)/R
    for k in range (N):
        yk = np.c_[Y[:,k]]
        zk = np.c_[Z[:,k]]
        ak = np.c_[A[:,k]]
        dk = D[k].copy()
        byk = yk.T@yk
        Mk= np.diag( np.squeeze((1-dk)*onesL + dk*zk))
        T1 = Mk @ GradP @ ak
        numG = numG + T1.T @ Mk @ (Po@ak- yk) / byk / N
        denG = denG + T1.T @ T1 / byk / N
    alpha = np.max([0, float(np.squeeze(numG/denG))])
    P_est = Po - alpha * GradP
    P_est[P_est<0]=0
    P_est[np.isnan(P_est)]=0
    P_est[np.isinf(P_est)]=0
    P = P_est.copy() / np.sum(P_est, axis=0)
    return P,NUMERROR



def totalVariance(A,Y,P,Lambda,nu,tau,nRow,nCol,epsilon,maxiter):
    def afun(W, nu, Dh, Dv, Weight):
        return nu * (Dh.T @ (Dh @ W) + Dv.T @ (Dv @ W)) + Weight @ W
    
    def SoftTh(B, omega):
        return np.sign(B) * np.maximum(0, np.abs(B) - omega)
    N,K = A.shape
    Ww=A.T.copy()
    b1 = np.zeros((K,1))
    b2 = np.zeros((K,1))
    p = np.zeros((K,1))
    q = np.zeros((K,1))
    

    # Create the sparse matrix
    data = np.array([-np.ones(nCol), np.ones(nCol)])
    Dh = spdiags(data, np.array([0, 1]), nCol, nCol, format='csr').toarray()
    Dh[nCol-1,:]=0
    Dh = ssp.kron(Dh, ssp.eye(nRow), format='csr')
    
    Dv = spdiags(data, np.array([0, 1]), nRow, nRow, format='csr').toarray()
    Dv[nRow-1,:]=0
    Dv = ssp.kron(Dv, ssp.eye(nCol), format='csr')
    Jp=1e-8;
    
    min_eigenvalue = np.min(splin.eigvals(P.T @ P).real)
    sum_squares = np.sum(Y**2, axis=0)
    diag_values = (min_eigenvalue * Lambda) / sum_squares
    Weight = ssp.csr_matrix(diag_values)
    Wp=Ww.copy()
    conv_sb=np.zeros((N,maxiter))    
    for j in range(N):
        for i in range(maxiter):
            Ay = Lambda *  A[j, :, np.newaxis] - (  nu * Dh.T.dot(p - b1) + nu * Dv.T.dot(q - b2) )
            def matvec(x):
                return nu * (Dh.T @ (Dh @ x) + Dv.T @ (Dv @ x)) + Weight @ x

            def rmatvec(x):
                return matvec(x)

            linear_operator = ssp.linalg.LinearOperator((Ww.shape[0], Ww.shape[0]), matvec=matvec, rmatvec=rmatvec)

            X = ssp.linalg.lsqr(linear_operator, Ay.flatten(), atol=1e-15, btol=1e-15, iter_lim=5, x0=Ww[:, j].copy())
            Ww[:, j] = X[0].copy()

            p = SoftTh(Dh @ Ww[:, j].copy().reshape(-1, 1) + b1, tau / nu)
            q = SoftTh(Dv @ Ww[:, j].copy().reshape(-1, 1) + b2, tau / nu)

            b1 += Dh @ Ww[:, j].reshape(-1, 1) - p
            b2 += Dv @ Ww[:, j].reshape(-1, 1) - q
            J = np.linalg.norm(Wp[:, j] - Ww[:, j], 2)
            conv_sb[j,i] = np.abs(J - Jp) / Jp
            if (np.abs(J - Jp) / Jp) < epsilon:
                break
            Jp = J
            Wp[:, j] = Ww[:, j].copy()

    Ww[Ww < 0] = 0
    W = Ww.copy().T 
    W = W/ np.sum(W, axis=0, keepdims=True)
    return W,conv_sb




def NEBEAESC(Y=[], N=2, parameters=[],Po=[],oae=0):
    L, K = Y.shape
    initcond=6
    rho=1
    Lambda=1e-4
    nu=10
    epsilon=1e-2
    maxiter=20
    parallel=0
    display=0
    nRow=int(np.sqrt(K))
    nCol=int(np.sqrt(K))
    NUMERROR = 0
    ## Check consistency of imput args    
    if np.size(parameters) !=11:
        print('The length of parameters vector is not 11 !!')
        print('Default values of hyper-parameters are used instead')
    else:
        initcond=int(parameters[0])
        rho = parameters[1]
        Lambda = parameters[2]
        tau=parameters[3]
        nu=parameters[4]
        nRow=parameters[5]
        nCol=parameters[6]
        epsilon=parameters[7]
        maxiter=parameters[8]
        parallel=parameters[9]
        display=parameters[10]     
        if initcond != 1 and initcond != 2 and initcond != 3  and initcond != 4 and initcond != 5 and initcond != 6 and initcond != 7 and initcond != 8:
            print("The initialization procedure of endmembers matrix is 1,2,3,4,5 or 6!")
            print("The default value is considered!")
            initcond = 1
        if rho <0:
            print('The regularization weight rho cannot be negative');
            print('The default value is considered!');
            rho=0.1;
        if Lambda<0 or Lambda>=1:
            print('The similarity weight in abundances is limited to [0,1)');
            print('The default value is considered!');
            Lambda=1e-4;
        if tau<0:
            print('The total variance weight has to be positive');
            print('The default value is considered!');
            tau=0.1;
        if nu<0:
            print('The split Bregman weight has to be positive');
            print('The default value is considered!');
            nu=10;
        if nRow*nCol != K:
            print('The product nRow x nCol does not match the spatial dimension!!');
            print('The default value is considered!');
            nRow=int(np.sqrt(K));
            nCol=nRow;
        if epsilon<0 or epsilon>0.5:
            print('The threshold epsilon cannot be negative or >0.5');
            print('The default value is considered!');
            epsilon=1e-3;
        if maxiter<0 and maxiter<100:
            print('The upper bound maxiter cannot be negative or >100');
            print('The default value is considered!');
            maxiter=20;
        if parallel!=0 and parallel!=1:
            print('The parallelization parameter is 0 or 1');
            print('The default value is considered!');
            parallel=0;
        if display!=0 and display!=1:
            print('The display parameter is 0 or 1');
            print('The default value is considered!');
            display=0;
    
    if not len(Y):
        print("The measurement matrix Y has to be used as argument!!")
        return 0
    else:
        if type(Y) != np.ndarray:
            print("The measurements matrix Y has to be a matrix")
            return
    if L>K:
        print('The number of spatial measurements has to be larger to the number of time samples!')
        return
    
    if N<2:
        print('The order of the linear mixture model has to greater than 2!')
        print('The default value n=2 is considered!')
        N = 2
    else:
        N=int(N)
    if len(Po):
        if type(Po) != np.ndarray:
            print("The initial end-members Po must be a matrix !!")
            print("The initialization is considered by tVCA")
            initcond = 6
        else:
            if Po.shape[0] == Y.shape[0] and Po.shape[1] == N:
                initcond = 0
            else:
                print("The size of Po must be M x n!!")
                print("The initialization is VCA")
                initcond = 6
    if oae != 0 and oae != 1:
        print("The assignment of oae is incorrect!!")
        print("The initial end-members Po will be improved iteratively from a selected sample")
        oae = 0
    elif oae == 1 and initcond != 0:
        print("The initial end-members Po is not defined properly!")
        print("Po will be improved iteratively from a selected sample")
        oae = 0
    
    W = np.zeros((N,K))
    D = np.zeros((K,))
    
    ## Normalizacion
    mYm = np.sum(Y, 0, keepdims=True)
    Ym = Y / mYm
    NYm = np.linalg.norm(Ym, 'fro') 
    
    ## Selection of End-Members Matrix
    if initcond == 1 or initcond == 2:
        if initcond == 1:
            Po = np.zeros((L, 1))
            index = 1
            p_max = np.mean(Y, axis=1)
            Yt = Y.copy()
            Po[:, index-1] = p_max
        elif initcond == 2:
            index = 1
            Y1m = np.sum(abs(Y), 0)
            y_max = np.max(Y1m)
            Imax = np.argwhere(Y1m == y_max)[0][0]
            y_min = np.min(Y1m)
            I_min = np.argwhere(Y1m == y_min)[0][0]
            p_max = Y[:, Imax]
            p_min = Y[:, I_min]
            II = np.arange(1, K)
            condition = np.logical_and(II != II[Imax], II != II[I_min])
            II = np.extract(condition, II)
            Yt = Y[:, II-1]
            Po = p_max
            index += 1
            Po = np.c_[Po, p_min]
        while index < N:
            y_max = np.zeros((1, index))
            Imax = np.zeros((1, index), dtype=int)
            for j in range(index):
                if j == 0:
                    for i in range(index):
                        e1m = np.around(np.sum(Yt*np.tile(Po[:, i], [Yt.shape[1], 1]).T, 0) /
                                        np.sqrt(np.sum(Yt**2, 0))/np.sqrt(np.sum(Po[:, i]**2, 0)), 4)
                        y_max[j][i] = np.around(np.amin(abs(e1m)), 4)
                        Imax[j][i] = np.where(e1m == y_max[j][i])[0][0]
            ym_max = np.amin(y_max)
            Im_max = np.where(y_max == ym_max)[1][0]
            IImax = Imax[0][Im_max]
            p_max = Yt[:, IImax]
            index += 1
            Po = np.c_[Po, p_max]
            II = np.arange(1, Yt.shape[1]+1)
            II = np.extract(II != IImax+1, II)
            Yt = Yt[:, list(II-1)]
    if initcond == 3:
        UU, s, VV = np.linalg.svd(Ym.T, full_matrices=False)
        W = VV.T[:, :N]
        Po = W * np.tile(np.sign(W.T@np.ones((L, 1))).T, [L, 1])
    if initcond == 4:
        Yom = np.mean(Ym, axis=1)
        Yon = Ym - np.tile(Yom[:, np.newaxis], (1, Y.shape[1]))
        _, S, VV = np.linalg.svd(Yon.T, full_matrices=False)
        Yo_w = np.linalg.pinv(np.sqrt(S)) @ VV.T @ Ym
        _, V, _ = np.linalg.svd(np.dot(np.tile(np.sum(Yo_w * Yo_w, axis=0), (N, 1)) * Yo_w, Yo_w.T))
        W = VV @ np.sqrt(S) @ V[:N, :].T
        Po = W * np.tile(np.sign(np.dot(W.T, np.ones((L, 1)))), (1, N))
    if initcond == 5:
        Po = NFINDR(Ym,N)
    if initcond == 6:
        Po,_,_ = vca(Ym,N)
    if initcond == 7:
        Po = SVMAX(Ym, N)

    
    Po = np.where(Po < 0, 0, Po)
    Po = np.where(np.isnan(Po), 0, Po)
    Po = np.where(np.isinf(Po), 0, Po)
    mPo=Po.sum(axis=0,keepdims=True)
    P=Po/np.tile(mPo,(L,1))   
    
    iter = 1
    J=1e5
    Jp = 1e6
    conv_track=np.zeros(maxiter+1)
    conv_track[0]= (Jp-J)/Jp
    #conv_sb = np.zeros((N,maxiter+1))
    if display == 1:
        print("#################################")
        print("NEBEAE SC Unmixing")
        print(f"Model Order = {N}")
        if oae == 1:
            print("Only the abundances are estimated from Po")
        elif oae == 0 and initcond == 0:
            print("The end-members matrix is initialized externally by matrix Po")
        elif oae == 0 and initcond == 1:
            print("Po is constructed based on the maximum cosine difference from mean measurement")
        elif oae == 0 and initcond == 2:
            print("Po is constructed based on the maximum and minimum energy, and largest difference from them")
        elif oae == 0 and initcond == 3:
            print("Po is constructed based on the PCA selection + Rectified Linear Unit")
        elif oae == 0 and initcond == 4:
            print("Po is constructed based on the ICA selection (FOBI) + Rectified Linear Unit")
        elif oae ==  0 and initcond == 5:
            print("Po is contructed based N-FINDR")
        elif oae == 0  and initcond == 6:
            print("Po is constructed based VCA")
        elif oae == 0  and initcond == 7:
            print("Po is constructed based SVMAX")
    while np.abs((Jp-J)/Jp) >= epsilon and iter<=maxiter and oae==0 and NUMERROR==0: 
    #while iter<=maxiter and oae==0 and NUMERROR==0:
        A, NUMERROR =abundance(Y, Ym, P, W, D, Lambda, NUMERROR)
        W, conv_sb = totalVariance(A, Ym, P, Lambda, nu, tau, nRow, nCol, epsilon, maxiter)
        D = probanonlinear(Y, Ym, P, A)
        Pp=P.copy()
        if NUMERROR == 0:
            P, NUMERROR = endmember(Y, Ym, Pp, A, D, rho, NUMERROR)
        Jp = J
        J = np.linalg.norm(Ym - ( (1-D)* (P @ A)) - (D * (P @ A * Y)), 'fro')
        conv_track[iter]= np.abs((Jp-J)/Jp)
        if J>Jp:
            P=Pp.copy()
            if display == 1:
                print(f"Number of iteration = {iter}")
                print(f"Percentage Estimation Error = {100 * J / NYm} %")
                break
        if display == 1:
            print(f"Number of iteration = {iter}")
            print(f"Percentage Estimation Error = {100 * J / NYm} %")
        iter += 1
    if NUMERROR==0 and oae==1:
        J=1e5
        Jp = 1e6
        D = np.zeros((K,))
        iter=1
        while np.abs((Jp-J)/Jp) >= epsilon and iter<=maxiter:
            A, NUMERROR =abundance(Y, Ym, P, W, D, Lambda, NUMERROR)
            W,conv_sb = totalVariance(A, Ym, P, Lambda, nu, tau, nRow, nCol, epsilon, maxiter)
            D = probanonlinear(Y, Ym, P, A)
            Jp = J
            J = np.linalg.norm(Ym - ( (1-D)* (P @ A)) - (D * (P @ A * Y)), 'fro')
            iter+=1
            if display==1:
                print(f"Number of iteration = {iter}")
                print(f"Percentage Estimation Error = {100 * J / NYm} %")
    S=mYm.T
    AA=A*mYm
    Yh = (1-D)*(P@AA) + D.T*(P@AA)*Y
    Ds = probanonlinear(Y,Y,P,A);
    if NUMERROR==1:
        print('Please revise the problem formulation, not reliable results')
        P=A=W=Ds=S=Yh=conv_track=conv_sb=[]
    
    return P,A,W,Ds,S,Yh,conv_track,conv_sb