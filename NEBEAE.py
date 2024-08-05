#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 11:05:55 2022

@author: nicolasmendoza
"""
import time
import numpy as np
from scipy import linalg
from scipy.sparse.linalg import svds
import scipy.linalg as splin

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
    return Ae,indice,Yp

def performance(fn):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        #print(f'Function {fn.__name__} took {t2-t1} s')
        return t2 - t1, result
    return wrapper

class NEBEAE:
    def __init__(self, Z=[], n=[], Po=[],initcond=1, epsilon=1e-3, maxiter=20, downsampling=0, parallel=0, display=0, oae=0):
        self.numerr=0
        self.oae=0
        self.rho=0.1
        self.Lambda = 0
        
        if not len(Z):
            print("The measurement matrix Y has to be used as argument!!")
            return 0
        else:
            if type(Z) != np.ndarray:
                print("The measurements matrix Y has to be a matrix")
                return 0
            else:
                self.Z=Z
        if n<2:
            print('The order of the linear mixture model has to greater than 2!')
            print('The default value n=2 is considered!')
            self.nperfiles=2
        else:
            self.nperfiles=int(n)
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
        if 0 > downsampling > 1:
            print("The downsampling factor cannot be negative or >1")
            print("The default value is considered!")
            self.downsampling = 0
        else:
            self.downsampling =downsampling
        if parallel != 0 and parallel != 1:
            print("The parallelization parameter is 0 or 1")
            print("The default value is considered!")
            self.parallel = 0
        else:
            self.parallel =parallel
        if display != 0 and display != 1:
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
                if Po.shape[0] == Z.shape[0] and Po.shape[1] == self.nperfiles:
                    self.initcond = 0
                    self.Po = Po
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
        
        Mbandas, Nopixeles = self.Z.shape
        if Mbandas > Nopixeles:
            print("The number of spatial measurements has to be larger to the number of time samples!")
            return 0
        #Npixeles = round(Nopixeles*(1-downsampling))
        
        #self.I = np.array(range(Nopixeles))
        #self.Is = np.random.choice(Nopixeles, Npixeles, replace=False)
        
        
        
        self.Y = self.Z.copy()#[:, self.Is-1].copy()
        # Normalization
        self.sumY = np.sum(self.Y, 0) #mYm
        self.sumZ = np.sum(self.Z, 0) #mYmo
        self.Ynorm = self.Y / self.sumY   #Ym
        self.Znorm = self.Z / self.sumZ #Ymo
        self.NYn = np.linalg.norm(self.Ynorm, 'fro') #NYm
        
        self.Mbandas, self.Npixeles = self.Ynorm.shape
        
        if self.oae ==0 :
            self.initializationEM()
        
        self.A = np.zeros((self.nperfiles,self.Npixeles))
        self.Yk=sum(self.Ynorm*self.Ynorm) 
        self.Dm = np.zeros((self.Npixeles,1))
        
    

    def initializationEM(self):
        if self.initcond == 1 or self.initcond == 2:
            if self.initcond == 1:
                self.Po = np.zeros((self.Mbandas, 1))
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
                II = np.arange(1, self.Npixeles)
                condition = np.logical_and(II != II[Imax], II != II[I_min])
                II = np.extract(condition, II)
                Yt = self.Y[:, II-1]
                self.Po = p_max
                index += 1
                self.Po = np.c_[self.Po, p_min]
            while index < self.nperfiles:
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
            UU, s, VV = np.linalg.svd(self.Ynorm.T, full_matrices=False)
            W = VV.T[:, :self.nperfiles]
            self.Po = W * np.tile(np.sign(W.T@np.ones((self.Mbandas, 1))).T, [self.Mbandas, 1])
        elif self.initcond == 4:
            Yom = np.mean(self.Ynorm, axis=1)
            Yon = self.Ym-np.tile(Yom, [self.Npixels, 1]).T
            UU, s, VV = np.linalg.svd(Yon.T, full_matrices=False)
            S = np.diag(s)
            Yo_w = np.linalg.pinv(linalg.sqrtm(S)) @ VV @ self.Ynorm
            V, s, u = np.linalg.svd((np.tile(sum(Yo_w * Yo_w), [self.Mbandas, 1]) * Yo_w) @ Yo_w.T, full_matrices=False)
            W = VV.T @ linalg.sqrtm(S)@V[:self.nperfiles, :].T
            self.Po = W*np.tile(np.sign(W.T@np.ones((self.Mbandas, 1))).T, [self.Mbandas, 1])
        elif self.initcond == 5:
            self.Po = NFINDR(self.Y,self.nperfiles)
        elif self.initcond == 6:
            self.Po,_,_ = vca(self.Y,self.nperfiles)
        else:
            self.P=self.Po.copy()
            
        self.Po = np.where(self.Po < 0, 0, self.Po)
        self.Po = np.where(np.isnan(self.Po), 0, self.Po)
        self.Po = np.where(np.isinf(self.Po), 0, self.Po)
        mPo=self.Po.sum(axis=0,keepdims=True)
        self.P=self.Po/mPo
   
    @performance
    def abun_par(self, Z, Y):
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
            byk = np.c_[Yk[k]]
            dk = np.c_[self.Dm[k]]
            deltakn = (1-dk)*np.ones((n,1))+dk*self.P.T@zk
            Pk = np.multiply(self.P,((1-dk)*np.ones((M,n))+dk*zk*np.ones((1,n))))
            bk = np.dot(Pk.T,yk)
            Go = np.dot(Pk.T,Pk)
            eGo, _ = np.linalg.eig(Go)
            eGo[np.isnan(eGo)]=1e6
            eGo[np.isinf(eGo)]=1e6
            lmin=np.amin(eGo)
            G=Go-np.eye(n)*lmin*self.Lambda
            Gi=np.linalg.pinv(G)
            #Gi = np.divide(np.eye(n),G)
    #Compute Optimal Unconstrained solutions
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
            self.A[:,k] = np.c_[ak].T
            
    @performance
    def prob_par(self, Z, Y):
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
        T1=EK - Y
        T2=EK - EK*Z
        T22=sum(T2*T2)
        T11=sum(T1*T2)
        dk=T11/T22
        return  np.minimum(1,dk.reshape((N,1)))
    
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
        dk=-num/den
        dk = np.minimum(1,dk.reshape((N,1)))
        #dk= np.zeros((dk.shape))
        return  dk


    @performance
    def endmember_par(self, Z, Y, Po):
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
        PAk=Po @ self.A
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
        
        cociente = float(numG/denG)
        alpha = np.max([0, cociente])
        # Compute the Stepest Descent Update of End-members Matrix
        P_est = Po - alpha * GradP
        P_est[P_est<0]=0
        P_est[np.isnan(P_est)]=0
        P_est[np.isinf(P_est)]=0
        self.P = P_est / np.sum(P_est, axis=0)
         
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
        d_Time = 0
        p_Time = 0
        #tic = time.time()
        if self.display == 1:
            print("#################################")
            print("NEBEAE Unmixing")
            print(f"Model Order = {self.nperfiles}")
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

        if self.oae == 1 and self.numerr==0:
            print('Only Optimal abundances Estimated')
            self.P=self.Po.copy()
            while (Jp-J)/Jp >= self.epsilon and ITER < self.maxiter:
                a_Time = self.abun_par(self.Z,self.Znorm)
                d_Time = self.prob_par(self.Z,self.Znorm)
                Jp= J
                a = np.multiply(np.tile((1-self.Dm).T, [self.Mbandas,1]),self.P@self.A)
                b = np.multiply(np.tile(self.Dm.T,[self.Mbandas,1]),np.multiply(self.P@self.A,self.Z))
                J = np.linalg.norm(self.Znorm-a-b,ord='fro')
                ITER += 1
            print('Percentage Estimation Error = ',100*J/self.NYn,'%')

        while (Jp-J)/Jp >= self.epsilon and ITER < self.maxiter and self.oae == 0 and self.numerr == 0:
            t_A=self.abun_par(self.Z, self.Ynorm)
            a_Time += t_A[0]
            t_D,self.Dm= self.prob_par(self.Z, self.Ynorm)
            d_Time += t_D
            Pp=self.P.copy()
            if self.numerr == 0:
                t_P=self.endmember_par(self.Z,self.Ynorm,Pp)
                p_Time += t_P[0]
            Jp = J
            a=np.multiply(np.tile((1-self.Dm).T,[self.Mbandas,1]),self.P@self.A)
            b=np.multiply(np.tile(self.Dm.T,[self.Mbandas,1]),np.multiply(self.P@self.A,self.Y))
            J = np.linalg.norm(self.Ynorm-a-b,ord='fro')
            if J > Jp:
                self.P = Pp.copy()
                break
            if self.display == 1:
                print(f"Number of iteration = {ITER}")
                print(f"Percentage Estimation Error = {(100*J)/self.NYn} %")
                print(f"Abundance estimation took {t_A}")
                print(f"Endmember estimation took {t_P}")
            ITER += 1
            
        # Ins = np.setdiff1d(self.I,self.Is)
        # #print(f"length of Ins = {len(Ins)}")
        # if self.downsampling > 0: #indices restantes del downsampling
            
        #     J = 1e5
        #     Jp = 1e6
        #     ITER = 1
            
        #     A_I=self.A.copy()
        #     Dm_I=self.Dm.copy()


        #     Ynorm_rest = self.Znorm[:,Ins]
        #     Y_rest = self.Z[:,Ins]
        #     Mrest, Nrest = Ynorm_rest.shape
            
        #     #print(f"size Ynorm_rest={Mrest , Nrest}")
            
            
            
        #     self.A = np.zeros((self.nperfiles, Nrest))
        #     self.Yk=sum(Ynorm_rest*Ynorm_rest)
        #     self.Dm= np.zeros((Nrest,1))
            
            
            
        #     while (Jp-J)/Jp >= self.epsilon and ITER < self.maxiter:
        #         t_A= self.abun_par(Y_rest,Ynorm_rest)
        #         a_Time += t_A[0]
        #         t_D, self.Dm = self.prob_par(Y_rest,Ynorm_rest)
        #         d_Time += t_D
        #         Jp = J
        #         a=np.multiply(np.tile((1-self.Dm).T,[Mrest,1]),self.P@self.A)
        #         b=np.multiply(np.tile(self.Dm.T,[Mrest,1]),np.multiply(self.P@self.A,Y_rest))
        #         J = np.linalg.norm(Ynorm_rest-a-b,ord='fro')
        #         ITER +=1
        #     self.A=np.concatenate((A_I,self.A),axis=1)
        #     self.Dm=np.concatenate((Dm_I,self.Dm),axis=0)
        
        # if self.oae == 0:
        #     II= np.concatenate((self.Is,Ins))
        #     Index=np.argsort(II)
        #     self.A=self.A[:,Index]
        #     self.Dm=self.Dm[Index,:]
        
        S= self.sumZ.T
        AA=np.multiply(self.A,np.tile(self.sumZ,[self.nperfiles,1]))
        a=np.multiply(np.tile((1-self.Dm).T,[self.Mbandas,1]),self.P@AA)
        b = np.multiply(np.tile(self.Dm.T,[self.Mbandas,1]),self.P@AA)
        self.Yh = a + np.multiply(b,self.Z)
        t_D,self.Dm = self.probfinal(self.Z)
        return self.P, self.A, self.Dm, self.Yh, S, d_Time, a_Time, p_Time