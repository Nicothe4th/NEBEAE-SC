#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 12:15:56 2023

@author: jnmc
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from spectral_tiffs import read_stiff, read_mtiff
from sklearn import metrics
sys.path.insert(0,'/Users/jnmc/Documents/Tesis/implenentaciones pyton/A_V_clase')
import CNEBEAE_SC as nstv4
import CNEBEAE_v1 as nebeae
import CEBEAESTV_v1 as lstv
import CEBEAE_V1 as ebeae

import pandas as pd

def replace(X):
    # Calculate mean and standard deviation
    arr = X.copy()
    mean_val = np.mean(arr)
    std_dev = np.std(arr)
    # Define a threshold for identifying outliers (e.g., 3 times the standard deviation)
    threshold = 3 * std_dev
    # Identify outliers
    outliers = np.abs(arr - mean_val) > threshold
    # Print the number of outliers
    num_outliers = np.sum(outliers) / len(arr) * 100
    print(f"% of outliers: {num_outliers}")
    # Replace outliers with the closest limit value
    arr[outliers] = np.clip(arr[outliers], mean_val - threshold, mean_val + threshold)
    return arr

def color_2(matriz):
    k,l=matriz.shape
    A=np.zeros((k,l,3)).astype(np.uint8)
    for i in range(k):
        for j in range(l):
            p=matriz[i,j]
            if p==0:
                A[i,j,:]=[255,255,255]
            if p==1:
                A[i,j,:]=[255,0,0]
            if p==2:
                A[i,j,:]=[0,255,0]
            if p==3:
                A[i,j,:]=[0,0,255]
    return A

def maxab(A,n1,n2,n3,n4):
    n, K = A.shape
    if n != (n1 + n2 + n3 + n4):
        print('Error dimentions not match')
        return []
    abun = np.zeros((4,K))
    lim1 = range(n1)
    lim2 = range(n1,n1+n2)
    lim3 = range(n1+n2,n1+n2+n3)
    lim4 = range(n1+n2+n3,n)
    
    abun[0,:]=sum(A[lim1,:])
    abun[1,:]=sum(A[lim2,:])
    abun[2,:]=sum(A[lim3,:])
    abun[3,:]=sum(A[lim4,:])    
    eti = np.argmax(abun,axis=0,keepdims=True)
    return abun, eti
def metricas(y_true, y_pred,label):
    label_names= ['Artery','Specular reflection','Vein','Stroma']
    m = metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred,display_labels=label_names);
    cm = m.confusion_matrix
    tp = np.zeros((1,4))
    fn = np.zeros((1,4))
    tn = np.zeros((1,4))
    fp = np.zeros((1,4))
    for i in range(4):
        tp[0,i] = cm[i,i]
        fn[0,i] = sum(cm[i,:])- cm[i,i]
        fp[0,i] = sum(cm[:,i])- cm[i,i]
        tn[0,i]= cm.sum()-(tp[0,i] + fn[0,i] + fp[0,i])
    acc = ((tp+tn)/(tp+tn+fn+fp)).ravel()
    sens =( tp/(tp+fn)).ravel()
    spec = (tn/(tn+fp)).ravel()
    f1 = (2*tp/(2*tp+fp)).ravel() 
    met = pd.DataFrame({'Tissue':['Artery','Specular reflection','Vein','Stroma'],'accuracy':acc,'sensivility':sens,'specifity':spec,'f1_score':f1})
    met.set_index('Tissue',inplace=True)
    met=met.T
    print('metricas por clase: \n',met)
    print(label)
    print('exactitud general =', np.sum(np.diag(cm))/np.sum(cm))
    return cm,met

# Read and Reshape mask and images

ImageC, center_wavelengths, Image_rgbC, metadata = read_stiff("/Users/jnmc/Desktop/placenta/Placenta P007 - P030 red blue/P011.tif")
masks = read_mtiff("/Users/jnmc/Desktop/placenta/Placenta P007 - P030 red blue/P011, masks.tif")

downsampling = range(0, len(ImageC), 1)
ImageE=ImageC[downsampling,:,:]
Image = ImageE[:,downsampling,:]

Image_rgbE = Image_rgbC[downsampling,:,:]
Image_rgb = Image_rgbE[:,downsampling,:]
z=Image_rgb[:,:,0]

arteryC = np.asarray(masks['Artery'],int)
arteryE = arteryC[downsampling,:]
artery = arteryE[:,downsampling]

spe_refC = np.asarray(masks['Specular reflection'],int)*2
spe_refE = spe_refC[downsampling,:]
spe_ref = spe_refE[:,downsampling]

veinC= np.asarray(masks['Vein'],int)*3
veinE = veinC[downsampling,:]
vein = veinE[:,downsampling]

stromaC = np.asarray(masks['Stroma'],int)*4
stromaE = stromaC[downsampling,:]
stroma = stromaE[:,downsampling]

Ny,Nx,Nz=Image.shape;
Z=Image.reshape((Nx*Ny,Nz),order='F').T
Z=Z/sum(Z)
L,K=Z.shape;
gt = (artery + spe_ref + stroma + vein)
gt[np.where(gt>4)]= 0 
M=gt.reshape((-1,1),order='F');

# End-members Estiamtion

I1,_=np.where(M==1);
I2,_=np.where(M==2);
I3,_=np.where(M==3);
I4,_=np.where(M==4);

Z1 = Z[:,I1].copy()
Z2 = Z[:,I2].copy()
Z3 = Z[:,I3].copy()
Z4 = Z[:,I4].copy()

nn1 = nn3 = 1
nn4 =2
nn2 = 1

nl1 = nl3 = 1 
nl4 =2
nl2 = 1

meanZ1 = np.mean(Z1,axis=1, keepdims=True)
std1 = np.std(Z1,axis =1) 
meanZ2 = np.mean(Z2,axis=1, keepdims=True)
std2 = np.std(Z2,axis =1) 
meanZ3 = np.mean(Z3,axis=1, keepdims=True)
std3 = np.std(Z3,axis =1) 
meanZ4 = np.mean(Z4,axis=1, keepdims=True)
std4 = np.std(Z4,axis =1) 


init_cond = 6;
if nn1 == 1:
    P1 = meanZ1
else:
    P1data = nebeae.NEBEAE(Z1, n= nn1, initcond=init_cond)
    P1, _, _, _, _, _, _, _= P1data.evaluate(rho=0.1, Lambda=0)

if nn2 == 1:
    P2 = meanZ2
else:
    P2data = nebeae.NEBEAE(Z2, n= nn2, initcond=init_cond)
    P2, _, _, _, _, _, _, _= P2data.evaluate(rho=0.1, Lambda=0)
    
if nn3 == 1:
    P3 = meanZ3
else:
    P3data = nebeae.NEBEAE(Z3, n = nn3, initcond=init_cond)
    P3, _, _, _, _, _, _, _= P3data.evaluate(rho=0.4, Lambda=0.1) 
if nn4 == 1:
    P4 = meanZ4
else:
    P4data = nebeae.NEBEAE(Z4, n = nn4, initcond=init_cond)
    P4, _, _, _, _, _, _, _= P4data.evaluate(rho=0.1, Lambda=0) 
    
P = np.concatenate((P1,P2,P3,P4),axis=1)

if nl1 == 1:
    Pl1 = meanZ1
else:
    datosl1=ebeae.EBEAE(Yo = Z1, n= nl1, initcond=init_cond)
    Pt, results_l= datosl1.evaluate(rho=0.01, Lambda=0)
    Pl1, _, _, _ ,_ ,_ = results_l

if nl2 == 1:
    Pl2 = meanZ2
else:
    datosl2=ebeae.EBEAE(Yo = Z2, n= nl2, initcond=init_cond)
    Pt, results_l= datosl2.evaluate(rho=0.01, Lambda=0)
    Pl2, _, _, _ ,_ ,_ = results_l
    
if nl3 == 1:
    Pl3 = meanZ3
else:
    datosl3=ebeae.EBEAE(Yo = Z3, n= nl3, initcond=init_cond)
    Pt, results_l= datosl3.evaluate(rho=0.01, Lambda=0)
    Pl3, _, _, _ ,_ ,_ = results_l
if nl4 == 1:
    Pl4 = meanZ4
else:
    datosl4=ebeae.EBEAE(Yo = Z4, n= nl4, initcond=init_cond)
    Pt, results_l= datosl4.evaluate(rho=0.01, Lambda=0)
    Pl4, _, _, _ ,_ ,_ = results_l

Pl= np.concatenate((Pl1,Pl2,Pl3,Pl4),axis=1)
## Supervisded Unmixing
#  seting parameters
a=np.sum(Z,axis=0,keepdims=True);
Y=Z/Z.max();
Po=P/np.sum(P,axis=0,keepdims=True);
Pol=Pl/np.sum(Pl,axis=0,keepdims=True);

_,n=Po.shape; 
_,nl=Pol.shape;
mu = 1e-2
nu = 1e-3
tau = 1e-2
               

sc={'mu': mu, 'nu':nu, 'tau':tau, 'dimX':Nx, 'dimY':Ny}
L= 0.1
R= 0.2

# unmixing

##              NEBEAE-STV   
datos = nstv4.NEBEAE_STV(Y, Po=Po, n=n, maxiter=20, display=0, oae=1, sc=sc)
_, Anstv, Dnstv, Wnstv, Yhnstv, _, _, _, t_= datos.evaluate(rho=R,Lambda=L) 


##              NEBEAE
datosnebeae = nebeae.NEBEAE(Y, Po=Po, n=n, maxiter=20, display=0, oae=1)
_, An, Dn, Yhn, _, _, _, _= datosnebeae.evaluate(rho=R, Lambda=L) 


##              EBEAE-STV    
datosL=lstv.EBEAE_STV(Yo=Y, n=nl, Po=Pol, maxiter=20, oae=1, sc=sc)
PLstv, ALstv, AnLstv, WLstv, YhLstv = datosL.evaluate(rho=R, Lambda=L)
   
    
    
##              EBEAE
 
#Yhf, Af, Pf, Dgmlm =spxls_gmlm(Y,z, nl, Pol, 500,10)
# Pf, Af, Anf, Yhf ,_ ,_ = results_l    

# generating the abundance maps
Anstv_sum, Anstv_eti = maxab(Wnstv,nn1,nn2,nn3,nn4)   
An_sum, An_eti = maxab(An,nn1,nn2,nn3,nn4)  
Alsc_sum, Alsc_eti = maxab(WLstv, nl1,nl2,nl3,nl4)
#Al_sum, Al_eti = maxab(Af, nl1,nl2,nl3,nl4)

mapaAnsc=Anstv_eti.reshape(Ny,Nx,order='F')
mapaAn=An_eti.reshape(Ny,Nx,order='F')
mapaAlsc=Alsc_eti.reshape(Ny,Nx,order='F')
#mapaAl=Al_eti.reshape(Ny,Nx,order='F')

Dcs = replace(Dnstv).reshape(Ny,Nx,order='F')
D = replace(Dn).reshape(Ny,Nx,order='F')

print('-----------Metricas de rendimiento-----------')
print('-----------------Parametros------------------')

print('Lambda = ',L)
print('Rho = ',R)
print('SC = ',sc)
# Calculated errors
print('error en reconstrucción NEBEAE-SC = ', np.linalg.norm(Y - Yhnstv, 'fro') )
print('error en reconstrucción NEBEAE = ', np.linalg.norm(Y - Yhn, 'fro') )
print('error en reconstrucción EBEAE-SC = ', np.linalg.norm(Y - YhLstv, 'fro') )
#print('error en reconstrucción G-MLM = ', np.linalg.norm(Y - Yhf, 'fro') )

indices,_=np.nonzero(M[:])
etiquetas=M[indices].ravel().astype(int)-1
cm,met = metricas(etiquetas,Anstv_eti[0,indices],'NEBEAE-SC')
cm,met = metricas(etiquetas,An_eti[0,indices],'NEBAE')
cm,met = metricas(etiquetas,Alsc_eti[0,indices],'EBEAE-SC')
#cm,met = metricas(etiquetas,Al_eti[0,indices],'G-MLM')


## Ploting
x_values = np.arange(len(P1))
plt.figure(1)
plt.clf()

plt.plot(P1, 'r', label='Artery')
plt.fill_between(x_values, meanZ1.ravel() - std1, meanZ1.ravel() + std1,  alpha=0.3, color='r')


plt.plot(P2, 'g', label='Specular reflection')
plt.fill_between(x_values, meanZ2.ravel() - std2, meanZ2.ravel() + std2,  alpha=0.3, color='g')

plt.plot(P3, 'b', label='Vein')
plt.fill_between(x_values, meanZ3.ravel() - std3, meanZ3.ravel() + std3,  alpha=0.3, color='b')


plt.plot(P4, 'k', label='Stroma')
plt.fill_between(x_values, meanZ4.ravel() - std4, meanZ4.ravel() + std4,  alpha=0.3, color='k')


plt.legend()
plt.show()


plt.figure(2)
plt.subplot(231)
plt.title('(A) RGB Image', fontweight="bold", fontsize=10)
plt.imshow(Image_rgb,aspect='auto')
plt.axis('off')

plt.subplot(232)
plt.title('(B) Ground-Truth Map', fontweight="bold", fontsize=10)
plt.imshow(color_2(gt),aspect='auto')
plt.axis('off')

plt.subplot(233)
plt.title('(C) Classified by NEBEAE-SC', fontweight="bold", fontsize=10)
plt.imshow(color_2(mapaAnsc+1),aspect='auto')
plt.axis('off')

plt.subplot(234)
plt.title('(D) Classified by NEBEAE',fontweight="bold", fontsize=10)
plt.imshow(color_2(mapaAn+1),aspect='auto')
plt.axis('off')

plt.subplot(235)
plt.title('(E) Classified by EBEAE-SC',fontweight="bold", fontsize=10)
#Di=Dnstv.reshape(Ny,Nx,order='F')
plt.imshow(color_2(mapaAlsc+1),aspect='auto')
plt.axis('off')

# plt.subplot(236)
# plt.title('(F) Classified by G-MLM',fontweight="bold", fontsize=10)
# Di=Dnstv.reshape(Ny,Nx,order='F')
# plt.imshow(color_2(mapaAl+1),aspect='auto')
# plt.axis('off')


vmax=max(replace(Dnstv).max(),replace(Dn).max())
vmin=min(replace(Dnstv).min(),replace(Dn).min())

plt.tight_layout()
plt.show()

plt.figure(3)
plt.subplot(221)
plt.title('(A) RGB Image', fontweight="bold", fontsize=10)
plt.imshow(Image_rgb)
plt.axis('off')

plt.subplot(222)
plt.title('(B) Ground-Truth Map', fontweight="bold", fontsize=10)
plt.imshow(color_2(gt))
plt.axis('off')

plt.subplot(223)
plt.imshow(Dcs)
plt.title('(C) NEBEAE-SC', fontweight="bold", fontsize=10)
plt.colorbar()
plt.clim(vmin,vmax)
plt.axis('off')
plt.subplot(224)
plt.imshow(D)
plt.title('(D) NEBEAE', fontweight="bold", fontsize=10)
plt.colorbar()
plt.clim(vmin,vmax)
plt.axis('off')
plt.tight_layout()
plt.show()
