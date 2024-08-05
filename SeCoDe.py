#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 23:51:21 2024

@author: jnmc
"""

import numpy as np 
from sklearn.preprocessing import normalize
from numpy.fft import fft2, ifft2
from scipy.sparse.linalg import LinearOperator, cg
import mat73
import warnings
import time

def SPCLSU_ADMM(Y, A, maxiter):
    
    epsilon = 1e-6
    iter = 0
    l, N = Y.shape
    l, k = A.shape
    P = np.zeros((l, k))
    Q = np.zeros((k, N))
    lamda1 = np.zeros_like(P)
    lamda2 = np.zeros_like(Q)
    stop = False
    mu = 1e-3
    rho = 1.5
    mu_bar = 1E+6

    while not stop and iter < maxiter + 1:
        iter += 1
        X1 = A.T@A + mu*np.eye(A.shape[1])
        X2 = A.T @ Y + +mu * Q+ lamda2
        X = np.linalg.inv(X1)@X2
        A = (Y @ X.T + mu * P + lamda1) @ np.linalg.inv(X @ X.T + mu * np.eye(X.shape[0]))
        
        P = np.maximum(A - lamda1 / mu, 0)
        Q = np.maximum(X - lamda2 / mu, 0)
        lamda1 = lamda1 + mu * (P - A)
        lamda2 = lamda2 + mu * (Q - X)
        mu = np.minimum(mu * rho, mu_bar)
        r_P = np.linalg.norm(P - A, 'fro')
        r_Q = np.linalg.norm(Q - X, 'fro')
        if r_P < epsilon and r_Q < epsilon:
            stop = True
            break

    return A, X

def lowpass(s, lambda_val, npad=16):
    """
    Lowpass filter image and return low and high frequency components.

    Parameters:
    s (ndarray): Input image or 3D array of images.
    lambda_val (float): Regularization parameter controlling lowpass filtering.
    npad (int): Number of samples to pad at image boundaries.

    Returns:
    sl (ndarray): Lowpass component.
    sh (ndarray): Highpass component.
    """
    npad=int(npad)
    grv = np.array([-1, 1]).reshape((1, 2))
    gcv = np.array([[-1], [1]]).reshape((2, 1))
    
    pad_shape = tuple(map(int, np.array(s.shape[:2]) + 2 * npad))
    Gr = np.fft.fft2(grv, pad_shape)
    Gc = np.fft.fft2(gcv, pad_shape)
    A = 1 + lambda_val * np.conj(Gr) * Gr + lambda_val * np.conj(Gc) * Gc
    
    sp = np.pad(s, ((npad, npad), (npad, npad), (0, 0)), mode='symmetric')
    slp = np.fft.ifft2(np.fft.fft2(sp, axes=(0, 1)) / A[:, :, np.newaxis], axes=(0, 1)).real
    
    sl = slp[npad:-npad, npad:-npad, :]
    sh = s - sl
    
    return sl, sh

def vec(v):
    u = np.reshape(v, (-1, 1))  # Reshape v into a column vector
    return u

def shrink(v, lambda_val):
    if np.isscalar(lambda_val):
        u = np.sign(v) * np.maximum(0, np.abs(v) - lambda_val)
    else:
        u = np.sign(v) * np.maximum(0, np.abs(v) - lambda_val)
    return u
def zpad(v, sz):
    u = np.zeros((sz[0], sz[1], *v.shape[2:]), dtype=v.dtype)
    u[:v.shape[0], :v.shape[1], ...] = v

    return u

def bcrop(v, sz):
    if len(sz) <= 2:
        if len(sz) == 1:
            cs = (sz[0], sz[0])
        else:
            cs = sz
        u = v[:cs[0], :cs[1], ...]
    else:
        cs = np.max(sz, axis=1)
        u = np.zeros((cs[0], cs[1], *v.shape[2:]), dtype=v.dtype)
        for k in range(v.shape[2]):
            u[:sz[0, k], :sz[1, k], k] = v[:sz[0, k], :sz[1, k], k]

    return u
def defaultopts(opt):
    # Set default values for each field if missing
    if 'Verbose' not in opt:
        opt['Verbose'] = 0
    if 'MaxMainIter' not in opt:
        opt['MaxMainIter'] = 1000
    if 'AbsStopTol' not in opt:
        opt['AbsStopTol'] = 1e-6
    if 'RelStopTol' not in opt:
        opt['RelStopTol'] = 1e-4
    if 'L1Weight' not in opt:
        opt['L1Weight'] = 1
    if 'Y0' not in opt:
        opt['Y0'] = []
    if 'U0' not in opt:
        opt['U0'] = []
    if 'G0' not in opt:
        opt['G0'] = []
    if 'H0' not in opt:
        opt['H0'] = []
    if 'rho' not in opt:
        opt['rho'] = []
    if 'AutoRho' not in opt:
        opt['AutoRho'] = 0
    if 'AutoRhoPeriod' not in opt:
        opt['AutoRhoPeriod'] = 10
    if 'RhoRsdlRatio' not in opt:
        opt['RhoRsdlRatio'] = 10
    if 'RhoScaling' not in opt:
        opt['RhoScaling'] = 2
    if 'AutoRhoScaling' not in opt:
        opt['AutoRhoScaling'] = 0
    if 'sigma' not in opt:
        opt['sigma'] = []
    if 'AutoSigma' not in opt:
        opt['AutoSigma'] = 0
    if 'AutoSigmaPeriod' not in opt:
        opt['AutoSigmaPeriod'] = 10
    if 'SigmaRsdlRatio' not in opt:
        opt['SigmaRsdlRatio'] = 10
    if 'SigmaScaling' not in opt:
        opt['SigmaScaling'] = 2
    if 'AutoSigmaScaling' not in opt:
        opt['AutoSigmaScaling'] = 0
    if 'StdResiduals' not in opt:
        opt['StdResiduals'] = 0
    if 'XRelaxParam' not in opt:
        opt['XRelaxParam'] = 1
    if 'DRelaxParam' not in opt:
        opt['DRelaxParam'] = 1
    if 'LinSolve' not in opt:
        opt['LinSolve'] = 'SM'
    if 'MaxCGIter' not in opt:
        opt['MaxCGIter'] = 1000
    if 'CGTol' not in opt:
        opt['CGTol'] = 1e-3
    if 'CGTolAuto' not in opt:
        opt['CGTolAuto'] = 0
    if 'CGTolAutoFactor' not in opt:
        opt['CGTolAutoFactor'] = 50
    if 'NoBndryCross' not in opt:
        opt['NoBndryCross'] = 0
    if 'DictFilterSizes' not in opt:
        opt['DictFilterSizes'] = []
    if 'NonNegCoef' not in opt:
        opt['NonNegCoef'] = 0
    if 'ZeroMean' not in opt:
        opt['ZeroMean'] = 0

    return opt
def solvedbi_sm(ah, rho, b, c=None):
    a = np.conj(ah)

    if c is None:
        denominator = np.sum(ah * a, axis=2) + rho
        c = ah / denominator[:,:,np.newaxis]
    c_expanded = c[..., np.newaxis]
    cb = np.sum(c_expanded * b, axis=2, keepdims=True)

    # Element-wise multiplication
    cba = np.multiply(cb,a[:, :, :, np.newaxis])
    x = (b - cba) / rho
        
    return x
def solvemdbi_ism(ah, rho, b):
    # Conjugate of ah
    a = np.conj(ah)
    K = ah.shape[3]  # Number of blocks

    # Initialize arrays
    gamma = np.zeros(a.shape,dtype=a.dtype)
    delta = np.zeros([a.shape[0], a.shape[1], 1, a.shape[3]],dtype=a.dtype)
    alpha = a[:, :, :, 0] / rho
    beta = b / rho
    del b

    for k in range(K):
        gamma[:, :, :, k] = alpha.copy()
        delta[:, :, 0, k] = 1 + np.sum(ah[:, :, :, k] * gamma[:, :, :, k], axis=2)

        c = np.sum(ah[:, :, :, k] * beta, axis=2)
        d = c[:, :, np.newaxis] * gamma[:, :, :, k]
        beta = beta - d / delta[:, :, 0, k, np.newaxis]

        if k <= K - 2:
            alpha = a[:, :, :, k + 1] / rho
            for l in range(k + 1):
                c = np.sum(ah[:, :, :, l] * alpha, axis=2)
                d = c[:, :, np.newaxis] * gamma[:, :, :, l]
                alpha = alpha - d / delta[:, :, 0, l, np.newaxis]

    x = beta.copy()
    return x
def solvemdbi_cg(ah, rho, b, tol=1e-5, mit=1000, isn=None):
    # Conjugate of ah
    a = np.conj(ah)
    asz = [ah.shape[0], ah.shape[1], ah.shape[2]]

    # Define linear operators
    def Aop(u):
        return np.sum(ah * u[:,:,None], axis=2)

    def Ahop(u):
        return np.sum(a * u[:,:,:,None], axis=3)

    def AhAvop(u):
        return Ahop(Aop(u.reshape(asz))).flatten()

    # Suppress warnings
    wrn = np.seterr(all='ignore')

    # Call conjugate gradient method
    if isn is None:
        isn = np.zeros_like(b)

    xv, flg = cg(LinearOperator((np.prod(asz), np.prod(asz)), matvec=lambda u: AhAvop(u) + rho * u), b.flatten(), tol=tol, maxiter=mit, x0=isn.flatten())

    # Restore warnings
    np.seterr(**wrn)

    # Create status structure
    cgst = {'flg': flg}

    # Reshape solution
    x = xv.reshape(asz)

    return x, cgst
def checkopt(inopt, dfopt):
    """
    Check options structure for unrecognized fields

    Parameters:
    ----------
    inopt : dict
        Input options dictionary
    dfopt : dict
        Default options dictionary

    Returns:
    -------
    None

    """
    if inopt and ('NoOptionCheck' not in inopt or not inopt['NoOptionCheck']):
        ifnc = inopt.keys()
        for ifn in ifnc:
            if ifn != 'NoOptionCheck' and ifn not in dfopt:
                warnings.warn(f"Unknown option field '{ifn}'", UserWarning)
                
def cbpdndl_unmixing(D0, S, Sl, gt, A, HSI, lambda_val, opt=None):
    if opt is None:
        opt = {}
    checkopt(opt, defaultopts({}))
    opt = defaultopts(opt)

    # Setup status display for verbose operation
    hstr = 'Itn   Fnc       DFid      l1        Cnstr     r(X)      s(X)      r(D)      s(D) '
    sfms = '%4d %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e'
    nsep = 84
    if opt['AutoRho']:
        hstr += '     rho  '
        sfms += ' %9.2e'
        nsep += 10
    if opt['AutoSigma']:
        hstr += '     sigma  '
        sfms += ' %9.2e'
        nsep += 10

    if opt['Verbose'] and opt['MaxMainIter'] > 0:
        print(hstr)
        print('-' * nsep)
    

    if S.shape[2] > 1:
        xsz = [S.shape[0], S.shape[1], D0.shape[2], S.shape[2]]
        S = np.reshape(S, [S.shape[0], S.shape[1], 1, S.shape[2]])
    else:
        xsz = [S.shape[0], S.shape[1], D0.shape[2], 1]

    Nx = np.prod(xsz)
    Nd = np.prod(xsz[:2]) * D0.shape[2]
    cgt = opt['CGTol']
    
    if not opt['DictFilterSizes']:
        dsz = [D0.shape[0], D0.shape[1]]
    else:
        dsz = opt['DictFilterSizes']

    # Mean removal and normalization projections
    def Pzmn(x):
        return np.subtract(x, np.mean(np.mean(x, axis=0), axis=1, keepdims=True))

    def Pnrm(x):
        ss1 = np.sqrt(np.sum(np.sum(x**2,axis=0,keepdims=True),axis=1,keepdims=True))
        x1 = x/ss1; 
        #row_norms = np.where(row_norms == 0, 1, row_norms)
        return x1

    # Projection of filter to full image size and its transpose (zero-pad and crop respectively)
    def Pzp(x):
        return zpad(x, xsz[:2])

    def PzpT(x):
        return bcrop(x, dsz)

    if opt['ZeroMean']:
        def Pcn(x):
            return Pnrm(Pzp(Pzmn(PzpT(x))))
    else:
        def Pcn(x):
            return Pnrm(Pzp(PzpT(x)))

    tstart = time.time()

    # Project initial dictionary onto constraint set
    D = Pnrm(D0)
    
    # Compute signal in DFT domain
    Sf = fft2(S, axes=(0, 1))
    
    rho = opt['rho'];
    if not rho:
        rho = 50 * lambda_val + 1

    if opt['AutoRho']:
        asgr = opt['RhoRsdlRatio']
        asgm = opt['RhoScaling']
    sigma = opt['sigma']
    if not sigma:
        sigma = S.shape[3]

    if opt['AutoSigma']:
        asdr = opt['SigmaRsdlRatio']
        asdm = opt['SigmaScaling']

    optinf = {'itstat': [],'opt': opt}
    rx = np.inf
    sx = np.inf
    rd = np.inf
    sd = np.inf
    eprix = 0
    eduax = 0
    eprid = 0
    eduad = 0

    X = None
    if not opt['Y0']:
        Y = np.zeros(xsz, dtype=S.dtype)
    else:
        Y = opt['Y0']

    Yprv = Y.copy()
    if not opt['U0']:
        if not opt['Y0']:
            U = np.zeros(xsz, dtype=S.dtype)
        else:
            U = (lambda_val / rho) * np.sign(Y)
    else:
        U = opt['U0'].copy()

    Df = None

    if not opt['G0']:
        G = Pzp(D)
    else:
        G = opt['G0'].copy()

    Gprv = G.copy()

    if not opt['H0']:
        if not opt['G0']:
            H = np.zeros_like(G)
        else:
            H = G.copy()
    else:
        H = opt['H0'].copy()
    
    Gf=fft2(G, axes=(0,1))
    GSf = np.multiply(np.conj(Gf[:, :, :, np.newaxis]), Sf)
    k = 1
    while k <= opt['MaxMainIter'] and (rx > eprix or sx > eduax or rd > eprid or sd > eduad):
        Xf = solvedbi_sm(Gf, rho, GSf + rho * fft2(Y - U, axes=(0,1)))
        X = ifft2(Xf,axes=(0, 1)).real
    
        if opt['XRelaxParam'] == 1:
            Xr = X.copy()
        else:
            Xr = opt['XRelaxParam'] * X + (1 - opt['XRelaxParam']) * Y

        Y = shrink(Xr + U, (lambda_val / rho) * opt['L1Weight'])
    
        if opt['NonNegCoef']:
            Y[Y < 0] = 0

        if opt['NoBndryCross']:
            Y[-D0.shape[0] + 1:, :, :, :] = 0
            Y[:, -D0.shape[1] + 1:, :, :] = 0

        Yf = fft2(Y,axes=(0,1))
        YSf_mult = np.multiply(np.conj(Yf), Sf)
        # Sum along the fourth dimension (axis=3 in Python)
        YSf = np.sum(YSf_mult, axis=3)        
        U = U + Xr - Y

        nX = np.linalg.norm(X)
        nY = np.linalg.norm(Y)
        nU = np.linalg.norm(U)

        if opt['StdResiduals']:
            rx = np.linalg.norm(vec(X - Y))
            sx = np.linalg.norm(vec(rho * (Yprv - Y)))
            eprix = np.sqrt(Nx) * opt['AbsStopTol'] + max(nX, nY) * opt['RelStopTol']
            eduax = np.sqrt(Nx) * opt['AbsStopTol'] + rho * nU *opt['RelStopTol']
        else:
            rx = np.linalg.norm(vec(X - Y)) / max(nX, nY)
            sx = np.linalg.norm(vec(Yprv - Y)) / nU
            eprix = np.sqrt(Nx) * opt['AbsStopTol'] / max(nX, nY) + opt['RelStopTol']
            eduax = np.sqrt(Nx) * opt['AbsStopTol'] / (rho * nU) + opt['RelStopTol']

        Jl1 = np.sum(np.abs(vec(np.multiply(opt['L1Weight'], Y))))
        Yprv = Y.copy()
        
        
        if opt['LinSolve'] == 'SM':
            Df = solvemdbi_ism(Yf, sigma, YSf + sigma * fft2((G - H), axes=(0,1)))
        else:
            print('notSM')
            Df, cgst = solvemdbi_cg(Yf, sigma, YSf + sigma * fft2((G - H),axes=(0,1)),
                                    cgt, opt['MaxCGIter'], Df.flatten())

        D = ifft2(Df, axes=(0,1)).real
        if opt['LinSolve'] == 'SM':
            del Df

        if opt['DRelaxParam'] == 1:
            Dr = D.copy()
        else:
            Dr = opt['DRelaxParam'] * D + (1 - opt['DRelaxParam']) * G

        G = Pcn(Dr + H)
        Gf = fft2(G,axes=(0,1))
        GSf = np.conj(Gf)[:, :, :, np.newaxis] * Sf
        
        H = H + Dr - G
        del Dr
        nD = np.linalg.norm(D)
        nG = np.linalg.norm(G)
        nH = np.linalg.norm(H)

        if opt['StdResiduals']:
            rd = np.linalg.norm(vec(D - G))
            sd = np.linalg.norm(vec(sigma * (Gprv - G)))
            eprid = np.sqrt(Nd) * opt['AbsStopTol'] + max(nD, nG) * opt['RelStopTol']
            eduad = np.sqrt(Nd) * opt['AbsStopTol'] + sigma * nH * opt['RelStopTol']
        else:
            rd = np.linalg.norm(vec(D - G)) / max(nD, nG)
            sd = np.linalg.norm(vec(Gprv - G)) / nH
            eprid = np.sqrt(Nd) * opt['AbsStopTol'] / max(nD, nG) + opt['RelStopTol']
            eduad = np.sqrt(Nd) * opt['AbsStopTol'] / (sigma * nH) + opt['RelStopTol']


        if opt['CGTolAuto'] and (rd / opt['opt.CGTolFactor']) < cgt:
            cgt = rd / opt['opt.CGTolFactor']

        Jcn = np.linalg.norm(vec(Pcn(D) - D))

        Gprv = G.copy()
        GYr = np.real(ifft2((Gf[:, :, :, np.newaxis] * Yf), axes=(0, 1)))
        
        GYfs = np.sum(Gf[:, :, :, np.newaxis] * Yf, axis=2, keepdims=True)
        

        Jdf = np.sum(vec(np.abs(GYfs - Sf) ** 2)) / (2 * xsz[0] * xsz[1])
        Jfn = Jdf + lambda_val * Jl1

        GY = np.squeeze( np.real(ifft2(np.transpose(GYfs, (0, 1, 3, 2)),axes=(0,1))) )
        est_S = np.abs(GY + Sl)
        

        tk = time.time() - tstart

        optinf['itstat'].append([k, Jfn, Jdf, Jl1, rx, sx, rd, sd, eprix, eduax, eprid, eduad, rho, sigma, tk])

        if opt['Verbose']:
            dvc = [k, Jfn, Jdf, Jl1, Jcn, rx, sx, rd, sd]
            if opt['AutoRho']:
                dvc.append(rho)
            if opt['AutoSigma']:
                dvc.append(sigma)
            print(sfms % tuple(dvc))
        

        if opt['AutoRho']:
            if k != 1 and k % opt['AutoRhoPeriod'] == 0:
                if opt['AutoRhoScaling']:
                    rhomlt = np.sqrt(rx / sx)
                    if rhomlt < 1:
                        rhomlt = 1 / rhomlt
                    if rhomlt > opt['RhoScaling']:
                        rhomlt = opt['RhoScaling']
                else:
                    rhomlt = opt['RhoScaling']

                rsf = 1
                if rx > opt['RhoRsdlRatio'] * sx:
                    rsf = rhomlt
                if sx > opt['RhoRsdlRatio'] * rx:
                    rsf = 1 / rhomlt

                rho = rsf * rho
                U = U / rsf
        
        if opt['AutoSigma']:
            if k != 1 and k % opt['AutoSigmaPeriod'] == 0:
                if opt['AutoSigmaScaling']:
                    sigmlt = np.sqrt(rd / sd)
                    if sigmlt < 1:
                        sigmlt = 1 / sigmlt
                    if sigmlt > opt['SigmaScaling']:
                        sigmlt = opt['SigmaScaling']
                else:
                    sigmlt = opt['SigmaScaling']
    
                ssf = 1
                if rd > opt['SigmaRsdlRatio'] * sd:
                    ssf = sigmlt
                if sd > opt['SigmaRsdlRatio'] * rd:
                    ssf = 1 / sigmlt
    
                sigma = ssf * sigma
                H = H / ssf

        k += 1
    D = PzpT(G)

    optinf['runtime'] = time.time() - tstart
    optinf['Y'] = Y
    optinf['U'] = U
    optinf['G'] = G
    optinf['H'] = H
    optinf['lambda'] = lambda_val
    optinf['rho'] = rho
    optinf['sigma'] = sigma
    optinf['cg_tol'] = cgt
    if 'cgst' in locals():
        optinf['cgst'] = cgst

    if opt['Verbose'] and opt['MaxMainIter'] > 0:
        print('-' * nsep)

    return D, Y, GYr, est_S, optinf

def NMF_ADMM(Y, Z, alfa, beta, maxiter):
    epsilon = 1e-6
    iter = 0

    l, N = Y.shape
    k, _ = Z.shape

    P = np.zeros((l, k))
    Q = np.zeros(Z.shape)
    G = np.zeros(Z.shape)

    lamda1 = np.zeros(P.shape)
    lamda2 = np.zeros(Q.shape)
    lamda3 = np.zeros(Z.shape)

    X = Z.copy()

    stop = False
    mu = 1e-3
    rho = 1.5
    mu_bar = 1e6

    while not stop and iter < maxiter + 1:
        iter += 1
        A = np.linalg.solve((X @ X.T + mu * np.eye(X.shape[0])).T, (Y @ X.T + mu * P+ lamda1).T).T
        
        X = np.linalg.solve(A.T @ A + (2 * mu + beta) * np.eye(A.shape[1]), A.T @ Y + beta * Z + mu * Q + lamda2 + mu * G + lamda3)
        
        G = np.maximum(np.abs(X - lamda3 / mu) - (alfa / mu), 0) * np.sign(X - lamda3 / mu)
        
        P = np.maximum(A - lamda1 / mu, 0)
        Q = np.maximum(X - lamda2 / mu, 0)

        lamda1 = lamda1 + mu * (P - A)
        lamda2 = lamda2 + mu * (Q - X)
        lamda3 = lamda3 + mu * (G - X)

        mu = min(mu * rho, mu_bar)

        r_P = np.linalg.norm(P - A, 'fro')
        r_Q = np.linalg.norm(Q - X, 'fro')
        r_G = np.linalg.norm(G - X, 'fro')

        if r_P < epsilon and r_Q < epsilon and r_G < epsilon:
            stop = True
            break
    
    return A, X

def secode(Y,P_0,h,w,Lambda,opt):
    h=int(h)
    w=int(w)
    _, A_0 = SPCLSU_ADMM(Y,P_0,100);
    A_0 = A_0 / np.sum(A_0, axis=0, keepdims=True)

    P_0 = normalize(P_0, norm='l1', axis=0)
    A_0 = normalize(A_0, norm='l1', axis=0)
    
    data_D= mat73.loadmat('D.mat')
    D=data_D['D']
    p=12

    _, k = P_0.shape
    F = D.copy()
#    delta = 1e-16
#    Y_a = np.vstack((Y, delta * np.ones((1, h*w))))

    for iter in range(10):
        #MC-CSC
        X_3d = np.reshape(A_0.T, (h, w, k), order='F')
        npd = p/2;
        fltlmbd = 10;
        Xl, Xh = lowpass(X_3d, fltlmbd, npd)
        F, M, FMh, FM, optinf = cbpdndl_unmixing(F, Xh, Xl, A_0, P_0, Y, Lambda, opt)
        Z= np.reshape(FM, A_0.T.shape , order='F').T
        P_1, A_1 = NMF_ADMM(Y, Z, 1e-1, 0.5, 1000)
    # Normalizing Z
    P_1 = P_1  / np.sum(P_1,axis=0,keepdims=True)
    Af = Z / np.sum(Z, axis=0, keepdims=True)  
    Zf= P_1@Af
    Zf = Zf / np.sum(Zf, axis=0,keepdims=True)
    return P_1, Af, Zf