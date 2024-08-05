#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 12:05:46 2023

@author: jnmc
"""
import time
import numpy as np
from scipy import linalg
from scipy.sparse.linalg import svds
import scipy.linalg as splin
from scipy.sparse import diags
import scipy.sparse as ssp

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
        TestM[1:N,i]= Yr[:N-1,i]
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
                actualSample = TestM[1:N,k]
                TestM[1:N,k] = Yr[:N-1,i]
                volume = np.abs(np.linalg.det(TestM))
                if volume > actualVolume:
                    actualVolume = volume
                    IDX[0,k] = i
                else:
                    TestM[1:N,k]=actualSample
        it = it + 1
        v1 = v2
        v2 = actualVolume
    
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
    Ae[Ae < 0] = 0
    return Ae,indice,Yp

def performance(fn):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        #print(f'Function {fn.__name__} took {t2-t1} s')
        return t2 - t1, result
    return wrapper

class NEBEAE_STV:
    def __init__(self, Yo=[], n=[], Po=[], initcond=1, epsilon=1e-3, maxiter=20,parallel=0, display=0,  oae=0, sc=[]):
        self.normalization = 1
        self.Z=Yo
        self.numerr=0
        self.oae=0
        self.rho=0.1
        self.Lambda = 0
        self.sc_pass=0
        self.sc= sc
        self.Po=Po

        if not len(Yo):
            print("The measurement matrix Y has to be used as argument!!")
            return 0
        else:
            if type(Yo) != np.ndarray:
                print("The measurements matrix Y has to be a matrix")
                return 0
            else:
                self.Yo=Yo
        if n<2:
            print('The order of the linear mixture model has to greater than 2!')
            print('The default value n=2 is considered!')
            self.n=2
        else:
            self.n=int(n)
        if initcond != 1 and initcond != 2 and initcond != 3  and initcond != 4 and initcond != 5 and initcond != 6:
            print("The initialization procedure of endmembers matrix is 1,2,3,4,5 or 6!")
            print("The default value is considered!")
            self.initcond = 1
        else:
            self.initcond = int(initcond)        

        if epsilon < 0 or epsilon > 0.5:
            print("The threshold epsilon can't be negative or > 0.5")
            print("The default value is considered!")
            self.epsilon = 1e-3
        else:
            self.epsilon = epsilon
        if maxiter < 0 and maxiter < 100:
            print("The upper bound maxiter can't be negative or >100")
            print("The default value is considered!")
            self.maxiter = 20
        else:
            self.maxiter = maxiter


        if parallel != 0 and parallel != 1:
            print("The parallelization parameter is 0 or 1")
            print("The default value is considered!")
            self.parallel = 0
        else:
            self.parallel =parallel
        if display != 0 and display != 1:
            print(f'display={display}')
            print("The display parameter is 0 or 1")
            print("The default value is considered")
            self.display = 0
        else:
            self.display = display
            
        if len(Po):
            if type(Po) != np.ndarray:
                print("The initial end-members Po must be a matrix !!")
                print("The initialization is considered by the maximum cosine difference from mean measurement")
                self.initcond = 1
            else:
                if Po.shape[0] == Yo.shape[0] and Po.shape[1] == self.n:
                    self.initcond = 0
                    self.P = Po.copy()
                else:
                    print("The size of Po must be M x n!!")
                    print("The initialization is considered based on the input dataset")
                    self.initcond = 1
        if oae != 0 and oae != 1:
            print("The assignment of oae is incorrect!!")
            print("The initial end-members Po will be improved iteratively from a selected sample")
            self.oae = 0
        elif oae == 1 and self.initcond != 0:
            print("The initial end-members Po is not defined properly!")
            print("Po will be improved iteratively from a selected sample")
            self.oae = 0
        elif oae == 1 and self.initcond == 0:
            self.oae = 1
        if type(sc) == dict:
            self.din=True
            self.sc_pass=1
        else:
            self.din=False
        if self.din:
            if type(sc) != dict:
                print("The inputs of spatial coherence are incorrect!!")
                print("---- Spatial consistency will not be considered ----")
                self.sc_pass=0
            elif len(sc) != 5:
                print("The number of inputs is incorrect!!")
                print("---- Spatial consistency will not be considered ----")
                self.sc_pass=0
            else:
                if sc["mu"]>1:
                    print("The value of mu for spatial coherence is unstable.")
                    print("The default value of mu=0.01 will be taken.")
                    self.sc["mu"]=0.01
                if sc["nu"]>1:
                    print("The value of nu for spatial coherence is unstable.")
                    print("The default value of nu=0.0001 will be taken.")
                    self.sc["nu"]=0.0001
                if sc["tau"]>1:
                    print("The value of nu for spatial coherence is unstable.")
                    print("The default value of tau=0.0001 will be taken.")
                    self.sc["tau"]=0.1
                if sc["dimX"]*sc["dimY"] != Yo.shape[1]:
                    print("X and Y dimensions do not correspond to the dimension of the input vector to be analyzed.!!")
                    print("---- Spatial consistency will not be considered ----")
                    self.sc_pass=0
        self.M, self.No= self.Yo.shape
        
        if self.M > self.No:
            print("The number of spatial measurements has to be larger to the number of time samples!")
            return 0
        
        #submuestreo
        self.N = self.No
        self.Y=self.Yo.copy()
        self.sumZ = np.sum(self.Z, 0,keepdims=True); #mYmo
        
        #normalización
        if self.normalization == 1:
            self.mYm = np.sum(self.Y, 0, keepdims=True)
            self.mYmo = np.sum(self.Yo, 0, keepdims=True)
        else:
            self.mYm = np.ones((1, self.N), dtype=int)
            self.mYmo = np.ones((1, self.No), dtype=int)
        self.Ym = self.Y / self.mYm
        self.Ymo = self.Yo / self.mYmo
        self.NYm = np.linalg.norm(self.Ym, 'fro')       
        
        if self.oae == 0:
            self.initializationEM()
        self.A = np.zeros((self.n,self.N))
        self.Wn = np.zeros([self.n,self.Y.shape[1]])
        self.Dm = np.zeros((self.N,1))
  
    def initializationEM(self):
        if self.initcond == 1 or self.initcond == 2:
            if self.initcond == 1:
                self.Po = np.zeros((self.M, 1))
                index = 1
                p_max = np.mean(self.Y, axis=1)
                Yt = self.Y
                self.Po[:, index-1] = p_max
            elif self.initcond == 2:
                index = 1
                Y1m = np.sum(abs(self.Y), 0)
                y_max = np.max(Y1m)
                Imax = np.argwhere(Y1m == y_max)[0][0]
                y_min = np.min(Y1m)
                I_min = np.argwhere(Y1m == y_min)[0][0]
                p_max = self.Y[:, Imax]
                p_min = self.Y[:, I_min]
                II = np.arange(1, self.N)
                condition = np.logical_and(II != II[Imax], II != II[I_min])
                II = np.extract(condition, II)
                Yt = self.Y[:, II-1]
                self.Po = p_max
                index += 1
                self.Po = np.c_[self.Po, p_min]
            while index < self.n:
                y_max = np.zeros((1, index))
                Imax = np.zeros((1, index), dtype=int)
                for j in range(index):
                    if j == 0:
                        for i in range(index):
                            e1m = np.around(np.sum(Yt*np.tile(self.Po[:, i], [Yt.shape[1], 1]).T, 0) /
                                            np.sqrt(np.sum(Yt**2, 0))/np.sqrt(np.sum(self.Po[:, i]**2, 0)), 4)
                            y_max[j][i] = np.around(np.amin(abs(e1m)), 4)
                            Imax[j][i] = np.where(e1m == y_max[j][i])[0][0]
                ym_max = np.amin(y_max)
                Im_max = np.where(y_max == ym_max)[1][0]
                IImax = Imax[0][Im_max]
                p_max = Yt[:, IImax]
                index += 1
                self.Po = np.c_[self.Po, p_max]
                II = np.arange(1, Yt.shape[1]+1)
                II = np.extract(II != IImax+1, II)
                Yt = Yt[:, list(II-1)]
        elif self.initcond == 3:
            UU, s, VV = np.linalg.svd(self.Ym.T, full_matrices=False)
            W = VV.T[:, :self.n]
            self.Po = W * np.tile(np.sign(W.T@np.ones((self.M, 1))).T, [self.M, 1])
        elif self.initcond == 4:
            Yom = np.mean(self.Ym, axis=1)
            Yon = self.Ym-np.tile(Yom, [self.N, 1]).T
            UU, s, VV = np.linalg.svd(Yon.T, full_matrices=False)
            S = np.diag(s)
            Yo_w = np.linalg.pinv(linalg.sqrtm(S)) @ VV @ self.Ym
            V, s, u = np.linalg.svd((np.tile(sum(Yo_w * Yo_w), [self.M, 1]) * Yo_w) @ Yo_w.T, full_matrices=False)
            W = VV.T @ linalg.sqrtm(S)@V[:self.n, :].T
            self.Po = W*np.tile(np.sign(W.T@np.ones((self.M, 1))).T, [self.M, 1])
        elif self.initcond == 5:
            self.Po = NFINDR(self.Y,self.n)
        elif self.initcond == 6:
            self.Po,_,_ = vca(self.Z,self.n)
        else:
            self.P=self.Po.copy()
            
        self.Po = np.where(self.Po < 0, 0, self.Po)
        self.Po = np.where(np.isnan(self.Po), 0, self.Po)
        self.Po = np.where(np.isinf(self.Po), 0, self.Po)
        mPo=self.Po.sum(axis=0,keepdims=True)
        self.P=self.Po/mPo   

    @performance   
    def abunTV(self, Z, Y):
        # Check arguments dimensions
        M, N = Y.shape
        n = self.P.shape[1]
        Yk=sum(Y*Y)
        if self.P.shape[0] != M:
            print("ERROR: the number of rows in Y and P does not match for abudances estimation")
            print('P.shape[0]=',self.P.shape[0])
            print('M=' ,M )
            self.numerr = 1
            return self.A
    
        for k in range(N):
            yk = np.c_[Y[:,k]]
            zk = np.c_[Z[:,k]]
            byk = Yk[k]
            dk = np.c_[self.Dm[k]]
            w = np.c_[self.Wn[:, k]]
            deltakn = (1-dk)*np.ones((n,1))+(dk*self.P.T@zk)

            Pk = np.multiply( self.P, ( (1-dk) * np.ones((M,n)) ) + ( dk * zk * np.ones((1,n)) ) )
            lgorrito = self.sc['mu'] * byk
            G = (Pk.T @ Pk) + lgorrito * np.eye((n))
            Ginv = np.linalg.pinv(G) 
         
            bk = (Pk.T @ yk)
            ck = lgorrito * w
            
            numk = deltakn.T @ Ginv @ (bk+ck) - 1
            denk = (deltakn.T @ Ginv @ deltakn) 
            dk = (numk / denk)* deltakn
            ak= Ginv @ (bk+ck-dk)

    #Check for Negative Elements
            if float(sum(ak >= 0)) != n:
                I_set = np.zeros((1, n))
                while float(sum(ak < 0)) != 0:
                    I_set = np.where(ak < 0, 1, I_set.T).reshape((1, n),order='F')
                    L = len(np.where(I_set == 1)[1])
                    Q = n+1+L
                    Gamma = np.zeros((Q, Q))
                    Beta = np.zeros((Q, 1))
                    Gamma[:n, :n] = G 
                    Gamma[:n, n] = (deltakn*byk).reshape((n,),order='F')
                    Gamma[n, :n] = deltakn.T
                    cont = 0
                    for i in range(n):
                        if I_set[:,i] != 0:
                            cont += 1
                            ind = i
                            Gamma[ind, n+cont] = 1
                            Gamma[n+cont, ind] = 1
                    Beta[:n, :] = bk
                    Beta[n, :] = 1
                    delta = np.linalg.solve(Gamma, Beta)
                    #print('delta =\n',delta)
                    ak = delta[:n]
                    ak = np.where(abs(ak) < 1e-9, 0, ak)
                    ak = ak/np.sum(ak)
            self.A[:,k] = np.c_[ak].T
            
    
    
    def SoftTh(self, B,lamb):
        return np.sign(B)*np.fmax(0,np.abs(B)-lamb*np.ones((len(B),1)))
    
    @performance
    def abun(self, Z, Y):
        # Check arguments dimensions
        M, N = Y.shape
        n = self.P.shape[1]
        Yk=sum(Y*Y)
        if self.P.shape[0] != M:
            print("ERROR: the number of rows in Y and P does not match for abudances estimation")
            print('P.shape[0]=',self.P.shape[0])
            print('M=' ,M )
            self.numerr = 1
            return self.A
    
        for k in range(N):
            yk = np.c_[Y[:,k]]
            zk = np.c_[Z[:,k]]
            byk = Yk[k]
            dk = np.c_[self.Dm[k]]
            deltakn = (1-dk)*np.ones((n,1))+(dk*self.P.T@zk)
            Pk = np.multiply(self.P, ((1-dk)*np.ones((M,n))+(dk*zk*np.ones((1,n)))))
            bk = np.dot(Pk.T,yk)
            Go = np.dot(Pk.T,Pk)
            eGo, _ = np.linalg.eig(Go)
            eGo[np.isnan(eGo)]=1e6
            eGo[np.isinf(eGo)]=1e6
            lmin=np.amin(eGo)
            G=Go-np.eye(n)*lmin*self.Lambda
            Gi=np.linalg.pinv(G)
            sigma = np.divide((deltakn.T@Gi@bk-1),(deltakn.T@Gi@deltakn))
            ak= Gi@ (bk-deltakn*sigma)
    #Check for Negative Elements
            if float(sum(ak >= 0)) != n:
                I_set = np.zeros((1, n))
                while float(sum(ak < 0)) != 0:
                    I_set = np.where(ak < 0, 1, I_set.T).reshape((1, n),order='F')
                    L = len(np.where(I_set == 1)[1])
                    Q = n+1+L
                    Gamma = np.zeros((Q, Q))
                    Beta = np.zeros((Q, 1))
                    Gamma[:n, :n] = G
                    Gamma[:n, n] = (deltakn*byk).reshape((n,),order='F')
                    Gamma[n, :n] = deltakn.T
                    cont = 0
                    for i in range(n):
                        if I_set[:,i] != 0:
                            cont += 1
                            ind = i
                            Gamma[ind, n+cont] = 1
                            Gamma[n+cont, ind] = 1
                    Beta[:n, :] = bk
                    Beta[n, :] = 1
                    delta = np.linalg.solve(Gamma, Beta)
                    ak = delta[:n]
                    ak = np.where(abs(ak) < 1e-9, 0, ak)
                    ak = ak/np.sum(ak)
            self.A[:,k] = np.c_[ak].T
            
        
    
    @performance
    def dn_abun(self, maxiter):
        
        dim,mn = self.A.shape
        b1=np.zeros((mn,1));
        W=self.A.T
        Wr=np.zeros(W.shape)
        b2=b1  
        p=b1   
        q=b1
        for j in range(dim):
            for i in range(maxiter):
                An=np.c_[self.A[j,:]]
                # Wn=np.c_[W[:,j]]
                Ay = self.sc['mu']* An + self.sc['nu'] * self.Dh.T@(p-b1)+self.sc['nu']*self.Dv.T@(q-b2)
                Wn=ssp.linalg.spsolve(self.tempval,Ay)
                
                a1=self.Dh@Wn.reshape([mn,1])
                a2=self.Dv@Wn.reshape([mn,1])
                
                p=self.SoftTh(a1+b1,self.sc['tau']/self.sc['nu']);
                q=self.SoftTh(a2+b2,self.sc['tau']/self.sc['nu']);
                b1=b1+a1-p;
                b2=b2+a2-q;                
                Wr[:,j]=Wn
        Wr[Wr<0]=0
        self.Wn = Wr.T.copy()
    
    @performance
    def endmember(self, Z, Y, Po):
        """
        P = endmember(Z,Y,P,A,rho,normalization)
        Estimation of Optimal End-members in Linear Mixture Model
        Input Arguments
        Z --> matrix of measurements
        Y -> matrix of normalized measurements
        P --> matrix of end-members
        A -->  matrix of abundances
        D --> vector of probabilities of nonlinear mixture
        rho = Weighting factor of regularization term
        Output Arguments
        P --> matrix of end-members
        Daniel U. Campos-Delgado Feb/2021
        Python version march/2022 JNMC
        """
        #Compute Gradient of Cost Function
        n, N = self.A.shape #n=number of endmembers N=pixels
        M, K = Y.shape #M=Bands K= pixels
        if Y.shape[1] != N:
            print("ERROR: the number of columns in Y and A does not match")
            self.numerr = 1
        GradP = np.zeros((M,n))
        R = sum(n-np.array(range(1, n),dtype='object'))
        Yk=sum(Y*Y)
        PAk= Po @ self.A
        # PAAk=PAk@A.T
        # dzk=Z*np.repeat(D.T,Z.shape[0],axis=0)
        for k in range(N):
            yk = np.c_[Y[:,k]]
            zk = np.c_[Z[:,k]]
            ak = np.c_[self.A[:,k]]
            pak= np.c_[PAk[:,k]]
            dk = np.c_[self.Dm[k]]
            byk = Yk[k]
            Mk=np.diag(((1-dk)*np.ones((M,1))+dk*zk).reshape((M,),order='F'))
            GradP = GradP- (Mk.T @ yk @ ak.T / byk) +( Mk.T @Mk @ pak @ ak.T) / byk   
        O = n * np.eye(n) - np.ones((n,n))
        GradP = GradP/N + self.rho * Po @ O/R
    
        #Compute Optimal Step in Update Rule
        numG = self.rho * np.trace(GradP @ O @ Po.T + Po @ O @ GradP.T)/R/2
        denG = self.rho * np.trace(GradP @ O @ GradP.T)/R
            
        for k in range (N):
            yk = np.c_[Y[:,k]]
            zk = np.c_[Z[:,k]]
            ak = np.c_[self.A[:,k]]
            pak= np.c_[PAk[:,k]]
            dk = np.c_[self.Dm[k]]
            byk = Yk[k]
            Mk=np.diag(((1-dk)*np.ones((M,1))+dk*zk).reshape((M,),order='F'))
            T1 = Mk @ GradP @ ak
            numG = numG + T1.T @ Mk @ (pak - yk) / byk / N
            denG = denG + T1.T @ T1 / byk / N
                
        alpha = np.max([0, numG/denG])
        # Compute the Stepest Descent Update of End-members Matrix
        P_est = Po - alpha * GradP
        P_est[P_est<0]=0
        P_est[np.isnan(P_est)]=0
        P_est[np.isinf(P_est)]=0
        self.P = P_est / np.sum(P_est, axis=0)
    
    @performance
    def endmember2(self,Y):
        M,K = Y.shape
        R = sum(self.n-np.array(range(1, self.n)))
        W = np.tile((1/K/sum(Y**2)), [self.n, 1]).T
        if Y.shape[1] != self.N:
            print("ERROR: the number of columns in Y and A does not match")
            self.numerr = 1
            return []
        
        o = (self.n * np.eye(self.n)-np.ones((self.n, self.n)))
        n1 = (np.ones((self.n, 1)))
        m1 = (np.ones((M, 1)))
    # Construct Optimal Endmembers Matrix
        T0 = (self.A @ (W*self.A.T)+self.rho*np.divide(o, R))
        rho=self.rho
        while 1/np.linalg.cond(T0, 1) < 1e-6:
            rho = rho/10
            T0 = (self.A @ (W*self.A.T)+rho*np.divide(o, R))
            if rho < 1e-6:
                print("Unstable numerical results in endmembers estimation, update rho!!")
                self.numerr = 1
        V = (np.eye(self.n) @ np.linalg.pinv(T0))
        T2 = (Y @ (W*self.A.T) @ V)
        if self.normalization == 1:
            T1 = (np.eye(M)-(1/M)*(m1 @ m1.T))
            T3 = ((1/M)*m1 @ n1.T)
            P_est = T1 @ T2 + T3
        else:
            P_est = T2
    
        # Evaluate and Project Negative Elements
        P_est = np.where(P_est < 0, 0, P_est)
        P_est = np.where(np.isnan(P_est), 0, P_est)
        P_est = np.where(np.isinf(P_est), 0, P_est)
    
        # Normalize Optimal Solution
        if self.normalization == 1:
            P_sum = np.sum(P_est, 0)
            self.P = P_est/np.tile(P_sum, [M, 1])
        else:
            self.P = P_est    
        
    @performance
    def prob(self, Z, Y):
        """
        D = probanonlinear(Z,Y,P,A,parallel)
        Estimation of Probability of Nonlinear Mixtures 
        Input Arguments
        Z --> matrix of measurements
        Y -> matrix of normalized measurements
        P --> matrix of end-members
        A -->  matrix of abundances
        parallel = implementation in parallel of the estimation
        Output Arguments
        D = Vector of probabilities of Nonlinear Mixtures
        Daniel U. Campos-Delgado February/2021
        Python version march/2022 JNMC
        """
        M, N = Y.shape
        dk = np.zeros((N,1))
        EK=self.P@self.A
        T1=Y-EK
        T2=EK - EK*Z
        T22=sum(T2*T2)
        T11=sum(T1*T2)
        dk=-T11/T22
        dk = np.minimum(1,dk.reshape((N,1)))
        #dk= np.zeros((dk.shape))
        return  dk
    
    @performance
    def probfinal(self, Z):
        """
        D = probanonlinear(Z,Y,P,A,parallel)
        Estimation of Probability of Nonlinear Mixtures 
        Input Arguments
        Z --> matrix of measurements
        Y -> matrix of normalized measurements
        P --> matrix of end-members
        A -->  matrix of abundances
        parallel = implementation in parallel of the estimation
        Output Arguments
        D = Vector of probabilities of Nonlinear Mixtures
        Daniel U. Campos-Delgado February/2021
        Python version march/2022 JNMC
        """
        M, N = Z.shape
        dk = np.zeros((N,1))
        Pwa = self.P@(self.A)
        #print(Pwa.shape)
        num1 = (Z-Pwa)
        num2 = (Pwa-(Pwa*Z))
        num = sum(num1*num2)
        den1 = Pwa-(Pwa*Z)
        den = sum(den1*den1)
        #print('num =',num)
        #print('den = ',den)
        dk= -num/den
        dk = np.minimum(1,dk.reshape((N,1)))
        #dk= np.zeros((dk.shape))
        return  dk
            
    
    def evaluate(self, rho=0.1, Lambda=0):
        if rho < 0:
            print("The regularization weight rho cannot be negative")
            print("The default value is considered!")
        else:
            self.rho = rho
        if Lambda < 0 or Lambda >= 1:
            print("The entropy weight lambda is limited to [0,1)")
            print("The default value is considered!")
        else:
            self.Lambda = Lambda
        
        ITER = 1
        J = 1e5
        Jp = 1e6
        a_Time = 0
        p_Time = 0
        w_Time = 0
        d_Time = 0
        tic = time.time()
        if self.display == 1:
            print("#################################")
            print("NEBEAE SC Unmixing")
            print(f"Model Order = {self.n}")
            if self.oae == 1:
                print("Only the abundances are estimated from Po")
            elif self.oae == 0 and self.initcond == 0:
                print("The end-members matrix is initialized externally by matrix Po")
            elif self.oae == 0 and self.initcond == 1:
                print("Po is constructed based on the maximum cosine difference from mean measurement")
            elif self.oae == 0 and self.initcond == 2:
                print("Po is constructed based on the maximum and minimum energy, and largest difference from them")
            elif self.oae == 0 and self.initcond == 3:
                print("Po is constructed based on the PCA selection + Rectified Linear Unit")
            elif self.oae == 0 and self.initcond == 4:
                print("Po is constructed based on the ICA selection (FOBI) + Rectified Linear Unit")
            elif self.oae ==  0 and self.initcond == 5:
                print("Po is contructed based N-FINDR")
            elif self.oae == 0  and self.initcond == 6:
                print("Po is constructed based VCA")
        if  self.sc_pass:
            self.Dh=diags([-1, 1], [0,1],shape=(self.sc['dimY'],self.sc['dimY'])).toarray()
            self.Dh[self.sc['dimY']-1,:]=0
            self.Dh=ssp.kron(self.Dh,np.eye(self.sc['dimX']),format='csr')
            self.Dv=diags([-1, 1], [0,1],shape=(self.sc['dimX'],self.sc['dimX'])).toarray()
            self.Dv[self.sc['dimX']-1,:]=0
            self.Dv=ssp.kron(np.eye(self.sc['dimY']),self.Dv,format='csr')
            self.tempval= self.sc['nu']*(self.Dh.T*self.Dh+self.Dv.T*self.Dv)+ self.sc['mu']*ssp.eye(self.Y.shape[1])        
            while (Jp-J)/Jp >= self.epsilon and ITER < self.maxiter  and self.numerr == 0:
                t_A = self.abunTV(self.Z, self.Ym)
                a_Time += t_A[0]
                t_D, self.Dm = self.prob(self.Z, self.Ym)
                d_Time = t_D
                Pp = self.P.copy()
                t_w = self.dn_abun(1)
                w_Time += t_w[0]
                if self.numerr == 0 and self.oae == 0:
                    t_P= self.endmember2(self.Ym)
                    p_Time += t_P[0]
                
                Jp = J
                a=np.multiply(np.tile((1-self.Dm).T,[self.M,1]),self.P@self.A)
                b=np.multiply(np.tile(self.Dm.T,[self.M,1]),np.multiply(self.P@self.A,self.Ym))
                J = np.linalg.norm(self.Ym-a-b, 'fro')

                if J > Jp and self.oae== 0:
                    self.P = Pp.copy()
                    #break
                
                if self.display == 1:
                    print(f"Number of iteration = {ITER}")
                    print(f"Percentage Estimation Error = {(100*J)/self.NYm} %")
                    print(f"Abundance estimation took {t_A}")
                    print(f"Endmember estimation took {t_P}")
                    print(f"Wn estimation took {t_w}")
                ITER += 1
            if self.numerr == 0:
                #t_A = self.abunTV(self.Z, self.Ymo)
                #t_w = self.dn_abun(1)
                self.An = self.A.copy() 
                
                #d_Time = t_D
                toc = time.time()
                elap_time = tic-toc
                if self.display == 1:
                    print(f"Elapsed Time = {elap_time} seconds") 
                AA=np.multiply(self.A,np.tile(self.sumZ,[self.n,1]))
                a=np.multiply(np.tile((1-self.Dm).T,[self.M,1]),self.P@AA)
                b = np.multiply(np.tile(self.Dm.T,[self.M,1]),self.P@AA)
                t_D, self.Dm = self.probfinal(self.Z)
                self.Yh = a + np.multiply(b,self.Z)
                S= self.sumZ.T
            else:
                print("Please review the problem formulation, not reliable results")
                self.P = np.array([])
                self.A = np.array([])
                self.An = np.array([])
                self.Yh = np.array([])
            return self.P, self.An, self.Dm, self.Wn, self.Yh, S, d_Time, a_Time, p_Time
        else:
            while (Jp-J)/Jp >= self.epsilon and ITER < self.maxiter and self.oae == 0 and self.numerr == 0:
                t_A = self.abun(self.Ym)
                a_Time += t_A[0]
                t_D,self.Dm = self.prob(self.Z, self.Ym)
                d_Time = t_D
                Pp = self.P.copy()
                if self.numerr == 0:
                    t_P = self.endmember(self.Z, self.Ym, Pp)
                    p_Time += t_P[0]
                Jp = J
                a=np.multiply(np.tile((1-self.Dm).T,[self.M,1]),self.P@self.A)
                b=np.multiply(np.tile(self.Dm.T,[self.M,1]),np.multiply(self.P@self.A,self.Y))
                J = np.linalg.norm(self.Ym-a-b, 'fro')
                if J > Jp:
                    self.P = Pp.copy()
                    break
                if self.display == 1:
                    print(f"Number of iteration = {ITER}")
                    print(f"Percentage Estimation Error = {(100*J)/self.NYm} %")
                    print(f"Abundance estimation took {t_A}")
                    print(f"Endmember estimation took {t_P}")
                ITER += 1
        
            if self.numerr == 0:    
                t_A = self.abun(self.Ym)        
                a_Time += t_A[0]
                t_D = self.prob(self.Z, self.Ym)
                d_Time = t_D[0]
                toc = time.time()
                elap_time = toc-tic
                if self.display == 1:
                    print(f"Elapsed Time = {elap_time} seconds")

                self.An = self.A.copy()
                self.A = self.A * np.tile(self.mYmo, [self.n, 1])
                self.Yh = self.P @ self.A
            else:
                print("Please review the problem formulation, not reliable results")
                self.P = np.array([])
                self.A = np.array([])
                self.An = np.array([])
                self.Yh = np.array([])
            return self.P, self.An, self.Dm, self.Wn, self.Yh, S, d_Time, a_Time, p_Time