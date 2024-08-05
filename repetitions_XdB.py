#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 15:02:35 2024

@author: nicolasmendoza
"""

import numpy as np 
import NEBEAE as nebeae
from NEBEAE_SC import NEBEAESC
import EBEAE_SC as lsc
from gmlm import spxls_gmlm
from SeCoDe import secode
from gtvMBO import gtvMBO_unmix
import UNSUBMM as unmix
import vnirsynth as VNIR
import matplotlib.pyplot as plt
import time
import pandas as pd
from aux import errors1, Anova
pd.set_option('display.float_format', '{:.2e}'.format)
from matplotlib.gridspec import GridSpec




nsamples = 120 # Size of the Squared Image nsamples x nsamples
#noise = 40 # Level in dB of Noise 40,35,30,25,20
# SNR = noise  
# PSNR = noise  
reps = 30






mutilde = 1e-2
nu = 1e-4
tau = 1e-5
R= 0.1

sc={'mu': mutilde, 'nu':nu, 'tau':tau, 'dimX':nsamples, 'dimY':nsamples}
L= mutilde
maxi = 20
m= 3


Lambda_secode = 1e-3 
opt = {
    "Verbose": 0,
    "MaxMainIter": 20,
    "rho": 100 * 1e-3 + 0.5,  # Assuming lambda is 1e-2
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
rho = 0.95
Lambda = 0.46
tau=3e4
nu=2e4
nRow=parameters=nsamples
nCol=parameters=nsamples
epsilon=parameters=1e-3
maxiter=parameters=20
parallel=parameters=0
display=parameters=0 

parameters = [initcond, rho, Lambda, tau, nu, nRow, nCol, epsilon, maxiter, parallel, display] 

ruido = [40,35,30,25,20]
for noise in ruido:
    tG = np.zeros((reps,)) #tiempo cómputo
    PG = np.zeros((reps,)) #error perfiles
    AG = np.zeros((reps,)) #error abundancias
    YG = np.zeros((reps,)) #error reconstrucción

    tN = np.zeros((reps,)) #tiempo cómputo
    PN = np.zeros((reps,)) #error perfiles
    AN = np.zeros((reps,)) #error abundancias
    YN = np.zeros((reps,)) #error reconstrucción


    tNstv = np.zeros((reps,))
    PNstv = np.zeros((reps,))
    ANstv = np.zeros((reps,))
    YNstv = np.zeros((reps,))

    tEstv = np.zeros((reps,))
    PEstv = np.zeros((reps,))
    AEstv = np.zeros((reps,))
    YEstv = np.zeros((reps,))

    tm = np.zeros((reps,)) #tiempo cómputo
    Pm = np.zeros((reps,)) #error perfiles
    Am = np.zeros((reps,)) #error abundancias
    Ym = np.zeros((reps,)) #error reconstrucción
    
    tmbo = np.zeros((reps,)) #tiempo cómputo
    Pmbo = np.zeros((reps,)) #error perfiles
    Ambo = np.zeros((reps,)) #error abundancias
    Ymbo = np.zeros((reps,)) #error reconstrucción

    tscd = np.zeros((reps,)) #tiempo cómputo
    Pscd = np.zeros((reps,)) #error perfiles
    Ascd = np.zeros((reps,)) #error abundancias
    Yscd = np.zeros((reps,)) #error reconstrucción
    
    print(f'Ruido = {noise}')
    SNR = int(noise)  
    PSNR = int(noise)  
    for i in range (reps):
        Yo, Po, Ao, Do = VNIR.VNIRsynthNLM(nsamples,SNR,PSNR, 4)
        a, b = np.shape(Po)
        z=np.reshape(Yo,(nsamples,nsamples,a))
        
        Pi,_,_ = nebeae.vca(Yo,3)
        Pi = Pi/np.tile(np.sum(Pi,axis=0,keepdims=True), (a, 1))
    #              NEBEAE-STV 
        start=time.perf_counter()
        Pnstv, Anstv, Wnstv, Dnstv ,Snstv, Yhnstv, conv_track, conv_sb = NEBEAESC(Yo, m, parameters, Pi, 0)
        print('n-sc')
        end1=time.perf_counter()
    

##              NEBEAE    
        datosn=nebeae.NEBEAE(Yo, Po= Pi, n=m, maxiter=20, display=0)
        Pn, An, Dn, Yhn, Sn, t_Dn, t_An, t_Pn= datosn.evaluate(rho=R,Lambda=L)
        end2=time.perf_counter()
        print('n')

##              EBEAE-STV    
        datosL=lsc.EBEAE_STV(Yo=Yo, n=m, Po=Pi, maxiter=20 ,sc=sc)
        PLstv, ALstv, AnLstv, WLstv, YhLstv = datosL.evaluate(rho=R, Lambda=L)
        end3=time.perf_counter()
        print('e')
    
##              GMLM
        Ygmlm, Agmlm, Pgmlm, Dgmlm = spxls_gmlm(Yo, z[:,:,0], m, Pi, 100, maxi)
        end4=time.perf_counter()
        print('gm')
    
#              UNSUBMM
        P, A, D, X, l = unmix.UNSUBMM(Yo, 3, Pi, 20)
        end5=time.perf_counter()
        print('un')
    

    #             SECODE
        print('secode')
        start=time.perf_counter()
        P_1, A_1, Z = secode(Yo,Pi,nsamples,nsamples,Lambda_secode,opt) 
        end6=time.perf_counter()
                #gtvMBO
        print('gtvMBO')
        A_MBO, S_MBO, Y_MBO, t_MBO =  gtvMBO_unmix(Yo,Pi,m,tol,nsamples,nsamples,sigma)  
        end7=time.perf_counter()
    
    
        Z= (P_1/np.sum(P_1,axis=0)) @ (A_1/np.sum(A_1,axis=0))
        Y_MBO = S_MBO @ A_MBO
        
        
        tNstv[i] = end1 - start
        tN[i] = end2 - end1
        tEstv[i] = end3 - end2
        tG[i] = end4 - end3
        tm[i] = end5-end4
        tscd[i] = end6 - end5
        tmbo[i] = end7 - end6
        YNstv[i], PNstv[i], ANstv[i] = errors1(Yo, Yhnstv, Ao, Wnstv, Po, Pnstv, m)
        YN[i],    PN[i],    AN[i]    = errors1(Yo, Yhn, Ao, An, Po, Pn, m)  
        YEstv[i], PEstv[i], AEstv[i] = errors1(Yo, YhLstv, Ao, WLstv, Po, PLstv, m)  
        YG[i],    PG[i],    AG[i]    = errors1(Yo, Ygmlm, Ao, Agmlm, Po, Pgmlm, m)
        Ym[i], Pm[i], Am[i] = errors1(Yo, X, Ao, A, Po, P, m)
        Yscd[i], Pscd[i], Ascd[i] = errors1(Yo/np.sum(Yo, axis=0), Z, Ao, A_1, Po, P_1, m)
        Ymbo[i], Pmbo[i], Ambo[i]= errors1(Yo/np.sum(Yo, axis=0), Y_MBO/np.sum(Y_MBO,axis=0), Ao, A_MBO, Po, S_MBO, m)  
   
    
   
    
   
    print(f'Ruido = {noise}')
    print('secode')
    print(f'YE = ${Yscd.mean():.4f} \pm {Yscd.std():.8f}$')
    print(f'AE = ${Ascd.mean():.4f} \pm {Ascd.std():.8f}$')
    print(f'PE = ${Pscd.mean():.4f} \pm {Pscd.std():.8f}$')
    print(f'time = ${tscd.mean():.4f} \pm {tscd.std():.8f}$')
    
    print('gtvMBO')
    print(f'YE = ${Ymbo.mean():.4f} \pm {Ymbo.std():.8f}$')
    print(f'AE = ${Ambo.mean():.4f} \pm {Ambo.std():.8f}$')
    print(f'PE = ${Pmbo.mean():.4f} \pm {Pmbo.std():.8f}$')
    print(f'time = ${tmbo.mean():.4f} \pm {tmbo.std():.8f}$') 

    



time = pd.DataFrame()
time['Nebeae-sc'] = pd.Series(tNstv)
time['Nebeae'] = pd.Series(tN)
time['Ebeae-sc'] = pd.Series(tEstv)
time['GMLM'] = pd.Series(tG)
time['UNSUBMM']= pd.Series(tm)
time['SeCoDe'] = pd.Series(tscd)
time['gtvMBO'] = pd.Series(tmbo)



EAb = pd.DataFrame()
EAb['Nebeae-sc'] = pd.Series(ANstv)
EAb['Nebeae'] = pd.Series(AN)
EAb['Ebeae-sc'] = pd.Series(AEstv)
EAb['G-MLM'] = pd.Series(AG)
EAb['UNSUBMM']= pd.Series(Am)
EAb['SeCoDe'] = pd.Series(Ascd)
EAb['gtvMBO'] = pd.Series(Ambo)

EEnd = pd.DataFrame()
EEnd['Nebeae-sc'] = pd.Series(PNstv)
EEnd['Nebeae'] = pd.Series(PN)
EEnd['Ebeae-sc'] = pd.Series(PEstv)
EEnd['G-MLM'] = pd.Series(PG)
EEnd['UNSUBMM']= pd.Series(Pm)
EEnd['SeCoDe'] = pd.Series(Pscd)
EEnd['gtvMBO'] = pd.Series(Pmbo)



ERec = pd.DataFrame()
ERec['Nebeae-sc'] = pd.Series(YNstv)
ERec['Nebeae'] = pd.Series(YN)
ERec['Ebeae-sc'] = pd.Series(YEstv)
ERec['G-MLM'] = pd.Series(YG)
ERec['UNSUBMM']= pd.Series(Ym)
ERec['SeCoDe'] = pd.Series(Yscd)
ERec['gtvMBO'] = pd.Series(Ymbo)








Anova(ERec,EEnd,EAb, time)

# Ar = np.zeros(np.shape(Ao))
# Ar[0,:]=Ao[2,:]
# Ar[1,:]=Ao[0,:]
# Ar[2,:]=Ao[1,:]

# fig = plt.figure(1, figsize=(10, 10))
# plt.clf()
# gs = GridSpec(8, 3, figure=fig)

# ax1 = fig.add_subplot(gs[0, 0])
# plt.title('End-member # 1',fontweight="bold", fontsize=10)
# plt.axis('off')
# plt.imshow(Ar[0,:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0],aspect='equal')
# ax2 = fig.add_subplot(gs[0, 1])
# plt.title('End-member # 2',fontweight="bold", fontsize=10)
# plt.axis('off')
# plt.imshow(Ar[1,:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0],aspect='equal')
# ax3 = fig.add_subplot(gs[0, 2])
# plt.title('End-member # 3',fontweight="bold", fontsize=10)
# plt.axis('off')
# plt.imshow(Ar[2,:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0],aspect='equal')

# ax4 = fig.add_subplot(gs[1, 0])
# plt.imshow(Wnstv[0,:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0],aspect='equal')
# plt.axis('off')
# ax5 = fig.add_subplot(gs[1, 1])
# plt.imshow(Wnstv[1,:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0],aspect='equal')
# plt.axis('off')
# ax6 = fig.add_subplot(gs[1, 2])
# plt.imshow(Wnstv[2,:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0],aspect='equal')
# plt.axis('off')

# ax7 = fig.add_subplot(gs[6, 0])
# plt.imshow(A_1[0,:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0],aspect='equal')
# plt.axis('off')
# ax8 = fig.add_subplot(gs[6, 1])
# plt.imshow(A_1[1,:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0],aspect='equal')
# plt.axis('off')
# ax9 = fig.add_subplot(gs[6, 2])
# plt.imshow(A_1[2,:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0],aspect='equal')
# plt.axis('off')


# ax7 = fig.add_subplot(gs[7, 0])
# plt.imshow(A_MBO[0,:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0],aspect='equal')
# plt.axis('off')
# ax8 = fig.add_subplot(gs[7, 1])
# plt.imshow(A_MBO[1,:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0],aspect='equal')
# plt.axis('off')
# ax9 = fig.add_subplot(gs[7, 2])
# plt.imshow(A_MBO[2,:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0],aspect='equal')
# plt.axis('off')


# ax13 = fig.add_subplot(gs[3, 0])
# plt.imshow(Agmlm[0,:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0],aspect='auto')
# plt.axis('off')
# ax14 = fig.add_subplot(gs[3, 1])
# plt.imshow(Agmlm[1,:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0],aspect='auto')
# plt.axis('off')
# ax15 = fig.add_subplot(gs[3, 2])
# plt.imshow(Agmlm[2,:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0],aspect='auto')
# plt.axis('off')

# ax16 = fig.add_subplot(gs[4, 0])
# plt.imshow(A[0,:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0],aspect='auto')
# plt.axis('off')
# ax17 = fig.add_subplot(gs[4, 1])
# plt.imshow(A[1,:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0],aspect='auto')
# plt.axis('off')
# ax18 = fig.add_subplot(gs[4, 2])
# plt.imshow(A[2,:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0],aspect='auto')
# plt.axis('off')

# ax10 = fig.add_subplot(gs[5, 0])
# plt.imshow(WLstv[0,:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0],aspect='auto')
# plt.axis('off')
# ax11 = fig.add_subplot(gs[5, 1])
# plt.imshow(WLstv[1,:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0],aspect='auto')
# plt.axis('off')
# ax12 = fig.add_subplot(gs[5, 2])
# plt.imshow(WLstv[2,:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0],aspect='auto')
# plt.axis('off')
# plt.colorbar()




#fig.suptitle('Estimated Abundance Maps With SNR/PSNR = 20 dB',fontweight="bold", fontsize=15)
# fig.text(0.5, .85, '(A) Grand-Truth', ha='center', fontsize=10, fontweight="bold")
# fig.text(0.5, .75, '(B) NEBEAE-SC', ha='center', fontsize=10, fontweight="bold")
# fig.text(0.5, .65, '(C) SeCoDe', ha='center', fontsize=10, fontweight="bold")
# fig.text(0.5, .60, '(D) gtvMBO', ha='center', fontsize=10, fontweight="bold")
# fig.text(0.5, .55, '(E) UNSUBMM', ha='center', fontsize=10, fontweight="bold")
# fig.text(0.5, .50, '(F) EBEAE-SC', ha='center', fontsize=10, fontweight="bold")
# fig.text(0.5, .44, '(G) SeCoDe', ha='center', fontsize=10, fontweight="bold")
# fig.text(0.5, .40, '(H) gtvMBO', ha='center', fontsize=10, fontweight="bold")
# plt.subplots_adjust(hspace=0.2, wspace=0.1)
# plt.show()

# plt.figure(figsize=(15, 5))
# ymin, ymax = 0, 4000
# plt.subplot(151)
# plt.hist(Do, bins=30, color='blue', alpha=0.5)
# plt.ylabel('Frequency', fontsize=10, fontweight="bold")
# plt.title('(A) Ground-Truth', fontsize=10, fontweight="bold")
# plt.ylim(ymin, ymax)
# plt.grid(True)


# plt.subplot(152)
# plt.hist(Dnstv, bins=30, color='green', alpha=0.5)
# plt.grid(True)
# plt.tick_params(axis='y', labelleft=False)  # Hide y-axis tick labels
# plt.title('(B) NEBEAE-SC', fontsize=10, fontweight="bold")
# plt.ylim(ymin, ymax)

# plt.subplot(153)
# plt.hist(Dn, bins=30, color='red', alpha=0.5)
# plt.xlabel('Value', fontsize=10, fontweight="bold")
# plt.title('(C) NEBEAE', fontsize=10, fontweight="bold")
# plt.tick_params(axis='y', labelleft=False)  # Hide y-axis tick labels
# plt.grid(True)
# plt.ylim(ymin, ymax)


# plt.subplot(154)
# plt.hist(Dgmlm, bins=30, color='orange', alpha=0.5)
# plt.title('(D) G-MLM', fontsize=10, fontweight="bold")
# plt.tick_params(axis='y', labelleft=False)  # Hide y-axis tick labels
# plt.ylim(ymin, ymax)
# plt.grid(True)

# plt.subplot(155)
# plt.hist(D, bins=30, color='purple', alpha=0.5)
# plt.title('(E) UNSUBMM ', fontsize=10, fontweight="bold")
# plt.tick_params(axis='y', labelleft=False)  # Hide y-axis tick labels
# plt.ylim(ymin, ymax)
# plt.grid(True)

# Displaying the plot
# plt.subplots_adjust(hspace=0.1, wspace=0.05)
# plt.show()