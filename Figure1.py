#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 10:22:58 2023

@author: jnmc
"""

import sys
sys.path.insert(0,'/Users/nicolasmendoza/Documents/implementaciones python/A_V_clase')
import CNEBEAE_v1 as nebeae
import CNEBEAE_SC as nsc

sys.path.insert(0,'/Users/nicolasmendoza/Documents/implementaciones python/bases_sintenticas')
import vnirsynth as VNIR
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

nsamples = 120  # Size of the Squared Image nsamples x nsamples
ruido = 20
SNR = ruido  # Level in dB of Gaussian Noise     SNR  = 45,50,55,60
PSNR = ruido  # Level in dB of Poisson/Shot Noise PSNR = 15,20,25,30
mu = 1e-5
nu = 1e-3
tau = 0.01

sc={'mu': mu, 'nu':nu, 'tau':tau, 'dimX':nsamples, 'dimY':nsamples}
L= 0.4
R= 0.5

m= 3
Yo, Po, Ao, Go = VNIR.VNIRsynthNLM(nsamples,SNR,PSNR, 4)

Pi,_,_ = nebeae.vca(Yo,3)
datosnstv=nsc.NEBEAE_STV(Yo, Po=Pi, n=m, maxiter=20, display=0, sc=sc)
Pnstv, Anstv, Dnstv, Wnstv, Yhnstv, Snstv, t_Dstv, t_Anstv, t_Pnstv= datosnstv.evaluate(rho=R,Lambda=L)

fig = plt.figure(1)
plt.clf()
gs = GridSpec(3, 3, figure=fig)

ax1 = fig.add_subplot(gs[0, 0])
plt.imshow(Ao[1,:].reshape((nsamples,nsamples)).T, aspect='equal')
plt.title('End-Member #1',fontweight="bold", fontsize=10)
plt.axis('off')

ax2 = fig.add_subplot(gs[0, 1])
plt.imshow(Ao[0,:].reshape((nsamples,nsamples)).T, aspect='equal')
plt.title('End-Member #2',fontweight="bold", fontsize=10)
plt.axis('off')

ax3 = fig.add_subplot(gs[0, 2])
plt.imshow(Ao[2,:].reshape((nsamples,nsamples)).T, aspect='equal')
plt.title('End-Member #3',fontweight="bold", fontsize=10)
plt.axis('off')


ax4 = fig.add_subplot(gs[1, 0])
plt.imshow(Anstv[0,:].reshape((nsamples,nsamples)).T,aspect='equal')
plt.axis('off')

ax5 = fig.add_subplot(gs[1, 1])
plt.imshow(Anstv[1,:].reshape((nsamples,nsamples)).T,aspect='equal')
#plt.title('Nominal abundances',fontweight="bold", fontsize=10)
plt.axis('off')

ax6 = fig.add_subplot(gs[1, 2])
plt.imshow(Anstv[2,:].reshape((nsamples,nsamples)).T,aspect='equal')
plt.axis('off')

ax7 = fig.add_subplot(gs[2, 0])
plt.imshow(Wnstv[0,:].reshape((nsamples,nsamples)).T,aspect='equal')
plt.axis('off')

ax8 = fig.add_subplot(gs[2, 1])
plt.imshow(Wnstv[1,:].reshape((nsamples,nsamples)).T,aspect='equal')
#plt.title('internal abundances',fontweight="bold", fontsize=10)
plt.axis('off')

ax9 = fig.add_subplot(gs[2, 2])
plt.imshow(Wnstv[2,:].reshape((nsamples,nsamples)).T,aspect='equal')
plt.axis('off')
fig.text(0.5, .63, 'Ground-Truth', ha='center', fontsize=10, fontweight="bold")
fig.text(0.5, .35, 'Nominal abundances', ha='center', fontsize=10, fontweight="bold")
fig.text(0.5, .08, 'Internal abundances', ha='center', fontsize=10, fontweight="bold")
plt.show()