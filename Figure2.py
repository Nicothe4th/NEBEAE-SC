#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 11:05:21 2023

@author: jnmc
"""

import sys
import numpy as np 
sys.path.insert(0,'/Users/nicolasmendoza/Documents/implementaciones python/bases_sintenticas')
import vnirsynth as VNIR
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec

nsamples = 120  # Size of the Squared Image nsamples x nsamples
ruido = 40
SNR = ruido  # Level in dB of Gaussian Noise     SNR  = 45,50,55,60
PSNR = ruido  # Level in dB of Poisson/Shot Noise PSNR = 15,20,25,30
Yo, Po, Ao, Go = VNIR.VNIRsynthNLM(nsamples,SNR,PSNR, 4)

fig = plt.figure(1)
plt.clf()
gs = GridSpec(2, 3, figure=fig)
ax1 = fig.add_subplot(gs[0, :])
plt.title('Spectral Signatures',fontweight="bold", fontsize=15)
plt.plot(Po)
plt.legend(["End-Member #1","End-Member #2","End-Member #3"])
plt.xlabel('Spectral Channel', fontsize=10, fontweight="bold")
plt.ylabel('Normalize Intensity', fontsize=10, fontweight="bold")
plt.grid(True)
ax2 = fig.add_subplot(gs[1, 0])
plt.axis('off')
plt.imshow(Ao[2,:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0],aspect='equal')
plt.title('End-Member #1',fontweight="bold", fontsize=10)
ax3 = fig.add_subplot(gs[1, 1])
plt.axis('off')
plt.imshow(Ao[0,:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0],aspect='equal')
plt.title('End-Member #2',fontweight="bold", fontsize=10)
ax4 = fig.add_subplot(gs[1, 2])
plt.axis('off')
plt.imshow(Ao[1,:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0],aspect='equal')
plt.title('End-Member #3',fontweight="bold", fontsize=10)
#fig.suptitle('Synthetic Database',fontweight="bold", fontsize=20)
plt.subplots_adjust(hspace=0.4, wspace=0.1)
fig.text(0.53, .43, 'Abundance Maps', ha='center', fontsize=15, fontweight="bold")
plt.show()
plt.tight_layout()