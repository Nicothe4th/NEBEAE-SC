#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 12:30:01 2023

@author: jnmc
"""
import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import time 


import NEBEAE as nebeae
from NEBEAE_SC import NEBEAESC
import EBEAE_SC as lstv
from gmlm import spxls_gmlm
from SeCoDe import secode
from gtvMBO import gtvMBO_unmix
import UNSUBMM as unmix





def errores_sinA(Yo, Y, Po, P, n):
    L, K = Y.shape
    Ep = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            Ep[i,j] = np.linalg.norm(Po[:, i] - P[:, j])            
    EY = np.linalg.norm(Yo - Y, 'fro')/np.linalg.norm(Yo,'fro')
    EP = sum(np.nanmin(Ep,axis=1)) 
    return EY, EP


data = scipy.io.loadmat('CupriteS1_R188.mat')
GT = scipy.io.loadmat('groundTruth_Cuprite_nEnd12.mat')


Y = data['Y']
Y = Y [1:,:]
nBand = data['nBand']
nCol = data['nCol']
nRow = data['nRow']
SlectBands = data['SlectBands']
M = GT['M']
nEnd = GT['nEnd']
slctBnds = GT['slctBnds']

# Z=Y/max(Y(:));
# n=12;
# Po=M(SlectBands',1:n);
# [L,K]=size(Z);
# k=1:L;

Z = Y/np.sum(Y,axis=0)
n = 12
P = np.squeeze(M[SlectBands.T,:])
L, K = Z.shape

mu = 0.001
nu = 0.0001
tau = 0.01

xpixels = 250
ypixels = 190


z=np.reshape(Z.T,(ypixels,xpixels,L))
z1=z[:,:,0]


sc={'mu': mu, 'nu':nu, 'tau':tau, 'dimX':xpixels, 'dimY':ypixels}
Lam= 0.001
R= 0.01

Lambda_secode = 1e-3 
opt = {
    "Verbose": 0,
    "MaxMainIter": 20,
    "rho": 100 * 1e-2 + 0.5,  # Assuming lambda is 1e-2
    "sigma": 5,
    "AutoRho": 1,
    "AutoRhoPeriod": 10,
    "AutoSigma": 1,
    "AutoSigmaPeriod": 10,
    "XRelaxParam": 1.8,
    "DRelaxParam": 1.8,
    "NonNegCoef": 1,
}

tol=1e-2
sigma = 5



initcond=int(6)
rho = 0.1
Lambda = 0.46
tau=3e4
nu=2e4
nRow=parameters=ypixels
nCol=parameters=xpixels
epsilon=parameters=1e-3
maxiter=parameters=20
parallel=parameters=0
display=parameters=0 

parameters = [initcond, rho, Lambda, tau, nu, nRow, nCol, epsilon, maxiter, parallel, display] 

Po,_,_ = nebeae.vca(Z,n)

start = time.perf_counter()






##              NEBEAE-STV 

Pnstv, Anstv, Wnstv, Dnstv ,Snstv, Yhnstv, conv_track, conv_sb = NEBEAESC(Z, n, parameters, Po, 0)
end1=time.perf_counter()

##              NEBAE    
datosn=nebeae.NEBEAE(Z, Po= Po, n=n, maxiter=20, display=0)
Pn, An, Dn, Yhn, Sn, t_Dn, t_An, t_Pn= datosn.evaluate(rho=R,Lambda=Lam)
end2=time.perf_counter()

##              EBEAE-STV    
datosL=lstv.EBEAE_STV(Yo=Z, n=n, Po=Po, maxiter=20 ,sc=sc)
PLstv, ALstv, AnLstv, WLstv, YhLstv = datosL.evaluate(rho=R, Lambda=Lam)
end3=time.perf_counter()
    
    
##              G-MLM

Ygmlm, Agmlm, Pgmlm, Dgmlm =spxls_gmlm(Z, z1, n, 100, 20)
end4=time.perf_counter()

Pu, Au, Du, Xu, lu = unmix.UNSUBMM(Z, n, 20)
end5=time.perf_counter()


# ##               secode
P_1, A_1, Z_1 = secode(Z,Po,nRow,nCol,Lambda_secode,opt)       
end6 =  time.perf_counter()
# ##             gtvMBO
A_MBO, S_MBO, Y_MBO, t_MBO =  gtvMBO_unmix(Z,Po,n,tol,nRow,nCol,sigma)  
end7 = time.perf_counter()


YNstv, PNstv = errores_sinA(Z, Yhnstv, Po, Pnstv, n)
YN,    PN    = errores_sinA(Z, Yhn, Po, Pn, n)  
YEstv, PEstv = errores_sinA(Z, YhLstv, Po, PLstv, n)  
YE, PE  = errores_sinA(Z, Ygmlm, Po, Pgmlm, n)
Yu, PU  = errores_sinA(Z, Xu, Po, Pu, n)
Ysec, Psec = errores_sinA(Z,Z_1,Po,P_1,n)
Ymbo, Pmbo = errores_sinA(Z,Y_MBO,Po,S_MBO,n)
print('-----------Metricas de rendimiento-----------')
print('-----------Cuprite-----------')
print('-----------------Parametros------------------')
print('Lambda = ',L)
print('Rho = ',R)
print('SC = ',sc)

print('-------------------NEBEAE-stv----------------')
print('error en reconstrucción = ', round(YNstv,6))
print('error en estimacion de endmembers = ', round(PNstv,6))     
print('tiempo de ejecución = ', end1-start,'s') 

print('---------------------NEBEAE-------------------')
print('error en reconstrucción = ', round(YN,6))
print('error en estimacion de endmembers = ', round(PN,6))     
print('tiempo de ejecución = ', end2-end1,'s') 

print('--------------------EBEAE-stv----------------')
print('error en reconstrucción = ', round(YEstv,6))
print('error en estimacion de endmembers = ', round(PEstv,6))     
print('tiempo de ejecución = ', end3-end2,'s') 

print('----------------------G-MLM-------------------')
print('error en reconstrucción = ', round(YE,6))
print('error en estimacion de endmembers = ', round(PE,6))     
print('tiempo de ejecución = ', end4-end3,'s') 

print('----------------------UNSUBMM-------------------')
print('error en reconstrucción = ', round(Yu,6))
print('error en estimacion de endmembers = ', round(PU,6))     
print('tiempo de ejecución = ', end5-end4,'s') 

print('----------------------SeCode-------------------')
print('error en reconstrucción = ', round(Ysec,6))
print('error en estimacion de endmembers = ', round(Psec,6))     
print('tiempo de ejecución = ', end6-end5,'s') 

print('----------------------gtvMBO-------------------')
print('error en reconstrucción = ', round(Ymbo,6))
print('error en estimacion de endmembers = ', round(Pmbo,6))     
print('tiempo de ejecución = ', end7-end6,'s') 



fig = plt.figure(1, figsize=(10, 10))
gs = GridSpec(2, 3, figure=fig)

ax1 = fig.add_subplot(gs[0, 0])
plt.title('End-member # 1',fontweight="bold", fontsize=10)
plt.axis('off')
#plt.imshow(.reshape((nsamples,nsamples)).T,extent = [0,100,100,0],aspect='equal')

ax2 = fig.add_subplot(gs[0, 1])
plt.title('NEBEAE-SC',fontweight="bold", fontsize=10)
plt.axis('off')
plt.imshow(Dnstv.reshape((ypixels,xpixels)).T,extent = [0,100,100,0],aspect='equal')


ax3 = fig.add_subplot(gs[0, 2])
plt.title('NEBEAE',fontweight="bold", fontsize=10)
plt.axis('off')
plt.imshow(Dn.reshape((ypixels,xpixels)).T,extent = [0,100,100,0],aspect='equal')

ax4 = fig.add_subplot(gs[1, 0])
plt.imshow(Dgmlm.reshape((ypixels,xpixels)).T,extent = [0,100,100,0],aspect='equal')
plt.title('G-MLM',fontweight="bold", fontsize=10)
plt.axis('off')

ax5 = fig.add_subplot(gs[1, 1])
plt.imshow(Du.reshape((ypixels,xpixels)).T,extent = [0,100,100,0],aspect='equal')
plt.title('UNSUBMM',fontweight="bold", fontsize=10)
plt.axis('off')








fig.suptitle('Estimated Non-linear Interaction Levels',fontweight="bold", fontsize=15)
#plt.subplots_adjust(hspace=0.2, wspace=0.1)
plt.show()


