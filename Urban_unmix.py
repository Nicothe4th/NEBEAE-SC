#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 12:38:53 2023

@author: jnmc
"""

import scipy
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.gridspec import GridSpec
from NEBEAE_SC import NEBEAESC
import NEBEAE as sc2
import EBEAE_SC as lstv
from gmlm import spxls_gmlm
import UNSUBMM as unmix
from aux import errors1, replace
from SeCoDe import secode
from gtvMBO import gtvMBO_unmix


data = scipy.io.loadmat('Urban_F210.mat')
UrbanRGB = plt.imread('UrbanRGB.png')

Y = data['Y']
nBand = np.squeeze(data['nBand'])
nCol = np.squeeze(data['nCol'])
nRow = np.squeeze(data['nRow'])
SlectBands = np.squeeze(data['SlectBands'])
maxValue = np.squeeze(data['maxValue'])



initcond=int(6)
rho = 0.1
Lambda = 0.46
tau=3e4
nu=2e4
# nRow=nRow
# nCol=nCol
epsilon=1e-3
maxiter=20
parallel=0
display=0 

parameters = [initcond, rho, Lambda, tau, nu, nRow, nCol, epsilon, maxiter, parallel, display] 


#nk = 4 # 4 , 5 , 6.
for nk in range(3):
    if nk == 0:
        Gt = scipy.io.loadmat('end4_groundTruth.mat')
    if nk == 1:
        Gt = scipy.io.loadmat('end5_groundTruth.mat')
    if nk == 2:
        Gt = scipy.io.loadmat('end6_groundTruth.mat')
        
    Ao = Gt['A']
    P = Gt['M']
    Pgt = P/sum(P)
    
    Z=np.squeeze(Y[SlectBands,:])/maxValue;
    L, K = Z.shape
    mu = 1e-4
    nu = 1e-7
    tau = 0.4
    
    _ ,n = P.shape
    
    xpixels = x = 307
    ypixels = y = 307
    z=np.reshape(Z,(xpixels,ypixels,L))
    
    sc={'mu': mu, 'nu':nu, 'tau':tau, 'dimX':xpixels, 'dimY':ypixels}
    L= 0.04
    R= 0.01
    
    
    
    
    Po,_,_ = sc2.vca(Z,n)
    Po = Po/sum(Po)
    
    
    Lambda_secode = 1e-2
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
    
    tol=1e-3
    sigma = 3
    
    start = time.perf_counter()
    



    
    ##              NEBEAE-STV 
    Pnstv, Anstv, Wnstv, Dnstv ,Snstv, Yhnstv, conv_track, conv_sb = NEBEAESC(Z, n, parameters, Po, 0)
    end1=time.perf_counter()
    
    # ##              NEBAE    
    datosc2=sc2.NEBEAE(Z, Po=Po, n=n, maxiter=20, display=0,  oae=0)
    Psc2, Asc2, Dsc2, Wsc2, Ysc2, Ssc2, tasc2,  tpsc2= datosc2.evaluate(rho=R,Lambda=L)
    end2=time.perf_counter()
    
    ##              EBEAE-STV    
    datosL=lstv.EBEAE_STV(Yo=Z, n=n, Po=Po, maxiter=20 ,sc=sc)
    PLstv, ALstv, AnLstv, WLstv, YhLstv = datosL.evaluate(rho=R, Lambda=L)
    end3=time.perf_counter()
        
        
    ##             GMLM
    Ygmlm, Agmlm, Pgmlm, Dgmlm =spxls_gmlm(Z,z[:,:,0],n, Po, 100, 10)
    end4=time.perf_counter()
    
    
    ##             UNSUBMM
    Pu, Au, Du, Xu, lu = unmix.UNSUBMM(Z, n, Po, 20)
    end5=time.perf_counter()
    ##            secode
    P_1, A_1, Z_1 = secode(Z,Po,nRow,nCol,Lambda_secode,opt)       
    end6=  time.perf_counter()
    ##             gtvMBO

    A_MBO, S_MBO, Y_MBO, t_MBO =  gtvMBO_unmix(Z/np.sum(Z,axis=0),Po,n,tol,nRow,nCol,sigma)  
    end7 = time.perf_counter()
    
    
    
   


    YNstv, PNstv, ANstv = errors1(Z, Yhnstv, Ao, Wnstv, Pgt, Pnstv, n)
    YN, PN, AN = errors1(Z, Ysc2, Ao, Wsc2, Pgt, Psc2, n)  
    YEstv, PEstv, AEstv = errors1(Z, YhLstv, Ao, WLstv, Pgt, PLstv, n)  
    YE, PE, AE = errors1(Z, Ygmlm, Ao, Agmlm, Pgt, Pgmlm, n)
    Yu, PU, AU  = errors1(Z, Xu, Ao, Au, Pgt, Pu, n)
    Ysec, Psec, Asec = errors1(Z,Z_1,Ao,A_1,Po,P_1,n)
    Y_MBO=S_MBO@A_MBO;
    Ymbo, Pmbo, Ambo = errors1(Z/np.sum(Z,axis=0),Y_MBO/np.sum(Y_MBO,axis=0),Ao/np.sum(Ao,axis=0),A_MBO,Po,S_MBO,n)

    fig = plt.figure(nk+1,tight_layout=True)
    plt.clf()
    gs = GridSpec(2, 3, figure=fig)
    
    ax1 = fig.add_subplot(gs[0, 0])
    plt.title('(A) Approximated RGB Image',fontweight="bold", fontsize=10)
    plt.axis('off')
    plt.imshow(UrbanRGB, aspect='equal')
    
    ax2 = fig.add_subplot(gs[0, 1])
    plt.title('(B) NEBEAE-SC',fontweight="bold", fontsize=10)
    plt.axis('off')
    plt.imshow((replace(Dnstv)).reshape((y,x)).T,aspect='equal')
    plt.colorbar()
    
    ax3 = fig.add_subplot(gs[0, 2])
    plt.title('(C) NEBEAE',fontweight="bold", fontsize=10)
    plt.axis('off')
    plt.imshow(replace(Dsc2).reshape((y,x)).T,aspect='equal')
    plt.colorbar()
    
    ax4 = fig.add_subplot(gs[1, 0])
    plt.imshow(Dgmlm.reshape((y,x)).T,aspect='equal')
    plt.title('(D) G-MLM',fontweight="bold", fontsize=10)
    plt.colorbar()
    plt.axis('off')
    
    
    ax5 = fig.add_subplot(gs[1, 1])
    plt.imshow(Du.reshape((y,x)).T,aspect='equal')
    plt.title('(E) UNSUBMM',fontweight="bold", fontsize=10)
    plt.colorbar()
    plt.axis('off')
    
    fig.suptitle(f'Urban Estimated \n Non-linear Interaction Levels with {4+nk} EM',fontweight="bold", fontsize=15)
    #plt.subplots_adjust(hspace=0.2, wspace=0.1)
    plt.show()

    print('-----------Perfomance Metrics-----------')
    print('-----------Urban-----------')
    print(f'_____________________n={n}___________________')
    
    
    print('-------------------NEBEAE-SC----------------')
    print('error en reconstrucción = ', round(YNstv,6))
    print('error en estimacion de abbundancias = ', round(ANstv,6))     
    print('error en estimacion de endmembers = ', round(PNstv,6))     
    print('tiempo de ejecución = ', end1-start,'s') 
    
    print('---------------------NEBEAE-------------------')
    print('error en reconstrucción = ', round(YN,6))
    print('error en estimacion de abundancias = ', round(AN,6))   
    print('error en estimacion de endmembers = ', round(PN,6))     
    print('tiempo de ejecución = ', end2-end1,'s') 
    
    print('--------------------EBEAE-SC----------------')
    print('error en reconstrucción = ', round(YEstv,6))
    print('error en estimacion de abundancias = ', round(AEstv,6)) 
    print('error en estimacion de endmembers = ', round(PEstv,6))     
    print('tiempo de ejecución = ', end3-end2,'s') 
    
    print('----------------------G-MLM-------------------')
    print('error en reconstrucción = ', round(YE,6))
    print('error en estimacion de abundancias = ', round(AE,6))
    print('error en estimacion de endmembers = ', round(PE,6))     
    print('tiempo de ejecución = ', end4-end3,'s') 
    
    print('----------------------UNSUBMM-------------------')
    print('error en reconstrucción = ', round(Yu,6))
    print('error en estimacion de abundancias = ', round(AU,6))  
    print('error en estimacion de endmembers = ', round(PU,6))     
    print('tiempo de ejecución = ', end5-end4,'s') 
 
    print('----------------------SeCode-------------------')
    print('error en reconstrucción = ', round(Ysec,6))
    print('error en estimacion de endmembers = ', round(Psec,6))  
    print('error en estimacion de abundancias = ', round(Asec,6))     
    print('tiempo de ejecución = ', end6-end5,'s') 
    print(f'number of end-members = {n}')
    print('----------------------gtvMBO-------------------')
    print('error en reconstrucción = ', round(Ymbo,6))
    print('error en estimacion de endmembers = ', round(Pmbo,6))     
    print('error en estimacion de abundancias = ', round(Ambo,6))     
    print('tiempo de ejecución = ', end7-end6,'s') 