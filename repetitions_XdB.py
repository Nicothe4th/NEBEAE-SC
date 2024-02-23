#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 15:02:35 2024

@author: nicolasmendoza
"""

import numpy as np 
import NEBEAE as nebeae
import NEBEAE_SC as nsc
import EBEAE_SC as lsc
from gmlm import spxls_gmlm
import UNSUBMM as unmix
import vnirsynth as VNIR
import matplotlib.pyplot as plt
import time
import pandas as pd
from aux import errors1, Anova
pd.set_option('display.float_format', '{:.2e}'.format)
from matplotlib.gridspec import GridSpec




nsamples = 120  # Size of the Squared Image nsamples x nsamples
noise = 40 # Level in dB of Noise 40,35,30,25,20
SNR = noise  
PSNR = noise  
reps = 30

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



mutilde = 1e-2
nu = 1e-4
tau = 1e-5
R= 0.1

sc={'mu': mutilde, 'nu':nu, 'tau':tau, 'dimX':nsamples, 'dimY':nsamples}
L= mutilde
maxi = 20
m= 3

for i in range (reps):
    Yo, Po, Ao, Do = VNIR.VNIRsynthNLM(nsamples,SNR,PSNR, 4)
    a, b = np.shape(Po)
    z=np.reshape(Yo,(nsamples,nsamples,a))
    start=time.perf_counter()
    Pi,_,_ = nebeae.vca(Yo,3)
    Pi = Pi/np.tile(np.sum(Pi,axis=0,keepdims=True), (a, 1))
#              NEBEAE-STV 
    datosnstv=nsc.NEBEAE_STV(Yo, Po=Pi, n=m, maxiter=20, display=0, sc=sc)
    Pnstv, Anstv, Dnstv, Wnstv, Yhnstv, Snstv, t_Dstv, t_Anstv, t_Pnstv= datosnstv.evaluate(rho=R,Lambda=L)
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
    
    tNstv[i] = end1 - start
    tN[i] = end2 - end1
    tEstv[i] = end3 - end2
    tG[i] = end4 - end3
    tm[i] = end5-end4
    
    YNstv[i], PNstv[i], ANstv[i] = errors1(Yo, Yhnstv, Ao, Wnstv, Po, Pnstv, m)
    YN[i],    PN[i],    AN[i]    = errors1(Yo, Yhn, Ao, An, Po, Pn, m)  
    YEstv[i], PEstv[i], AEstv[i] = errors1(Yo, YhLstv, Ao, WLstv, Po, PLstv, m)  
    YG[i],    PG[i],    AG[i]    = errors1(Yo, Ygmlm, Ao, Agmlm, Po, Pgmlm, m)
    Ym[i], Pm[i], Am[i] = errors1(Yo, X, Ao, A, Po, P, m)
    print('iter =',i)

time = pd.DataFrame()
time['Nebeae-sc'] = pd.Series(tNstv)
time['Nebeae'] = pd.Series(tN)
time['Ebeae-sc'] = pd.Series(tEstv)
time['GMLM'] = pd.Series(tG)

EAb = pd.DataFrame()
EAb['Nebeae-sc'] = pd.Series(ANstv)
EAb['Nebeae'] = pd.Series(AN)
EAb['Ebeae-sc'] = pd.Series(AEstv)
EAb['G-MLM'] = pd.Series(AG)

EEnd = pd.DataFrame()
EEnd['Nebeae-sc'] = pd.Series(PNstv)
EEnd['Nebeae'] = pd.Series(PN)
EEnd['Ebeae-sc'] = pd.Series(PEstv)
EEnd['G-MLM'] = pd.Series(PG)

ERec = pd.DataFrame()
ERec['Nebeae-sc'] = pd.Series(YNstv)
ERec['Nebeae'] = pd.Series(YN)
ERec['Ebeae-sc'] = pd.Series(YEstv)
ERec['G-MLM'] = pd.Series(YG)


time['UNSUBMM']= pd.Series(tm)
EEnd['UNSUBMM']= pd.Series(Pm)
ERec['UNSUBMM']= pd.Series(Ym)
EAb['UNSUBMM']= pd.Series(Am)


Anova(ERec,EEnd,EAb, time)

Ar = np.zeros(np.shape(A))
Ar[0,:]=Ao[2,:]
Ar[1,:]=Ao[0,:]
Ar[2,:]=Ao[1,:]

fig = plt.figure(1, figsize=(10, 10))
plt.clf()
gs = GridSpec(6, 3, figure=fig)

ax1 = fig.add_subplot(gs[0, 0])
plt.title('End-member # 1',fontweight="bold", fontsize=10)
plt.axis('off')
plt.imshow(Ar[0,:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0],aspect='auto')
ax2 = fig.add_subplot(gs[0, 1])
plt.title('End-member # 2',fontweight="bold", fontsize=10)
plt.axis('off')
plt.imshow(Ar[1,:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0],aspect='auto')
ax3 = fig.add_subplot(gs[0, 2])
plt.title('End-member # 3',fontweight="bold", fontsize=10)
plt.axis('off')
plt.imshow(Ar[2,:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0],aspect='auto')

ax4 = fig.add_subplot(gs[1, 0])
plt.imshow(Wnstv[0,:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0],aspect='auto')
plt.axis('off')
ax5 = fig.add_subplot(gs[1, 1])
plt.imshow(Wnstv[1,:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0],aspect='auto')
plt.axis('off')
ax6 = fig.add_subplot(gs[1, 2])
plt.imshow(Wnstv[2,:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0],aspect='auto')
plt.axis('off')

ax7 = fig.add_subplot(gs[2, 0])
plt.imshow(An[0,:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0],aspect='auto')
plt.axis('off')
ax8 = fig.add_subplot(gs[2, 1])
plt.imshow(An[1,:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0],aspect='auto')
plt.axis('off')
ax9 = fig.add_subplot(gs[2, 2])
plt.imshow(An[2,:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0],aspect='auto')
plt.axis('off')


ax13 = fig.add_subplot(gs[3, 0])
plt.imshow(Agmlm[0,:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0],aspect='auto')
plt.axis('off')
ax14 = fig.add_subplot(gs[3, 1])
plt.imshow(Agmlm[1,:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0],aspect='auto')
plt.axis('off')
ax15 = fig.add_subplot(gs[3, 2])
plt.imshow(Agmlm[2,:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0],aspect='auto')
plt.axis('off')

ax16 = fig.add_subplot(gs[4, 0])
plt.imshow(A[0,:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0],aspect='auto')
plt.axis('off')
ax17 = fig.add_subplot(gs[4, 1])
plt.imshow(A[1,:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0],aspect='auto')
plt.axis('off')
ax18 = fig.add_subplot(gs[4, 2])
plt.imshow(A[2,:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0],aspect='auto')
plt.axis('off')

ax10 = fig.add_subplot(gs[5, 0])
plt.imshow(WLstv[0,:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0],aspect='auto')
plt.axis('off')
ax11 = fig.add_subplot(gs[5, 1])
plt.imshow(WLstv[1,:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0],aspect='auto')
plt.axis('off')
ax12 = fig.add_subplot(gs[5, 2])
plt.imshow(WLstv[2,:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0],aspect='auto')
plt.axis('off')

fig.suptitle('Estimated Abundance Maps With SNR/PSNR = 40 dB',fontweight="bold", fontsize=15)
fig.text(0.5, .755, '(A) Grand-Truth', ha='center', fontsize=10, fontweight="bold")
fig.text(0.5, .625, '(B) NEBEAE-SC', ha='center', fontsize=10, fontweight="bold")
fig.text(0.5, .493, '(C) NEBEAE', ha='center', fontsize=10, fontweight="bold")
fig.text(0.5, .36, '(D) G-MLM', ha='center', fontsize=10, fontweight="bold")
fig.text(0.5, .227, '(E) UNSUBMM', ha='center', fontsize=10, fontweight="bold")
fig.text(0.5, .095, '(F) EBEAE-SC', ha='center', fontsize=10, fontweight="bold")
plt.subplots_adjust(hspace=0.2, wspace=0.1)
plt.show()

plt.figure(figsize=(15, 5))
ymin, ymax = 0, 4000
plt.subplot(151)
plt.hist(Do, bins=30, color='blue', alpha=0.5)
plt.ylabel('Frequency', fontsize=10, fontweight="bold")
plt.title('(A) Ground-Truth', fontsize=10, fontweight="bold")
plt.ylim(ymin, ymax)
plt.grid(True)


plt.subplot(152)
plt.hist(Dnstv, bins=30, color='green', alpha=0.5)
plt.grid(True)
plt.tick_params(axis='y', labelleft=False)  # Hide y-axis tick labels
plt.title('(B) NEBEAE-SC', fontsize=10, fontweight="bold")
plt.ylim(ymin, ymax)

plt.subplot(153)
plt.hist(Dn, bins=30, color='red', alpha=0.5)
plt.xlabel('Value', fontsize=10, fontweight="bold")
plt.title('(C) NEBEAE', fontsize=10, fontweight="bold")
plt.tick_params(axis='y', labelleft=False)  # Hide y-axis tick labels
plt.grid(True)
plt.ylim(ymin, ymax)


plt.subplot(154)
plt.hist(Dgmlm, bins=30, color='orange', alpha=0.5)
plt.title('(D) G-MLM', fontsize=10, fontweight="bold")
plt.tick_params(axis='y', labelleft=False)  # Hide y-axis tick labels
plt.ylim(ymin, ymax)
plt.grid(True)

plt.subplot(155)
plt.hist(D, bins=30, color='purple', alpha=0.5)
plt.title('(E) UNSUBMM ', fontsize=10, fontweight="bold")
plt.tick_params(axis='y', labelleft=False)  # Hide y-axis tick labels
plt.ylim(ymin, ymax)
plt.grid(True)

# Displaying the plot
plt.subplots_adjust(hspace=0.1, wspace=0.05)
plt.show()