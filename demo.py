
"""
Created on Tue Apr 28 12:06:36 2015

@author: Zhe Gan
"""

import numpy as np
import scipy.io

np.random.seed(1234)

data = scipy.io.loadmat('mnist_part.mat')
N, Nx, Ny, X = data['N'], data['Nx'], data['Ny'], data['X']

N = np.array(N)[0]
Nx = np.array(Nx)[0]
Ny = np.array(Ny)[0]

X = X.reshape(Nx,Ny,1,N)
nd = np.array([7,7])
Mp = np.array([3,3])
K = 20
Xn = 1
L = 1
nx = np.array([28,28])

burnin = 200
num = 50
space = 6
CFA_result = []



# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 14:02:10 2015

@author: Zhe Gan
"""
import numpy as np
import numpy.random as npr
import time

def Gibbs(X,nd,K,Mp,nx,N,Xn,L,burnin,num,space,CFA_result):
    """ The L-th layer, loading data. """
    
   #Mp2 = Mp[0]*Mp[1]
    nx2 = nx[0]*nx[1]
    ns = nx-nd+1
    nXn = nx[0]*nx[1]*Xn
    #ns2 = ns[0]*ns[1]
    
    #if L == 2:
    #    rfft_D = np.fft.fft2(CFA_result["D"],(28,28),axes=(0,1))
    
    # set parameters
    c0,d0,g0,h0,D_a0,D_b0 = 1e-6,1e-6,1e0,1e-3,1e0,1e-3
    
    # initialization
    D = np.zeros((nd[0],nd[1],Xn,K))
    D_alpha = npr.gamma(D_a0+0.5, 1/(D_b0+0.5*D**2))
    phi = nx2*np.ones(N)
    a = 1.0
    pia0 = a/K
    pib0 = 1-pia0
    pai = npr.beta(pia0,pib0*np.ones(K))
    b = np.ones((K,N))
    P = np.zeros((Mp))
    P[0,0] = 1
    Z = np.ones(((nx-nd)/Mp+1))
    Z = np.tile(Z,N)
    Z = np.kron(Z,P).reshape(ns[0]+Mp[0]-1,ns[1]+Mp[1]-1,N)
    Z = Z[:-1-Mp[0]+2,:-1-Mp[1]+2,:]
    Ind = (np.abs(Z-1.0)<= 1e-6).nonzero() # ??
    S_alpha = np.zeros((ns[0],ns[1],N))
    S = npr.randn(ns[0],ns[1],K,N)
    S = Z.reshape(ns[0],ns[1],1,N)*S
    bSZ = b.reshape(1,1,K,N)*S
    
    fft_D = np.fft.fft2(D,(nx[0],nx[1]),axes = (0,1))
    fft_SZ = np.fft.fft2(bSZ,(nx[0],nx[1]),axes = (0,1))
    res=X;
    iternum = 0; 
    
    begin = time.time()
    while (iternum < burnin+num*space):
        iternum = iternum + 1
        
        midphi = phi.reshape(1,1,1,N)
        midphi3 = phi.reshape(1,1,N)
        
        for kk in range(K):
            """ compute X_{-n} """            
            ASZ = np.real(np.fft.ifft2(fft_D[:,:,:,[kk]]*fft_SZ[:,:,[kk],:]))
            midres=res+ASZ
            
            """ Sample b """
            Sk = S[:,:,kk,:]
            P = -(res**2 - midres**2)*midphi/2
            P = np.sum(P,axis=(0,1,2))+(np.log(pai[kk])-np.log(1-pai[kk]))
            b[kk,:] = npr.rand(N) <np.exp(P)
            fft_SZ[:,:,kk,:] = np.fft.fft2(b[kk,:].reshape(1,1,N)*Sk,(nx[0],nx[1]),axes=(0,1))
            ASZ = np.real(np.fft.ifft2(fft_D[:,:,:,[kk]]*fft_SZ[:,:,[kk],:]))
            res=midres-ASZ
            
            """ Sample S """
            sumD = np.sum(D[:,:,:,kk]**2)
            S_alpha[Ind]= npr.gamma(g0 + 0.5,1./(h0+ 0.5*(Sk[Ind]**2)))
            
            if np.sum(b[kk,:]) > 0:
                S_sig = 1/(sumD*midphi3*Z*b[kk,:].reshape(1,1,N) + S_alpha + 1e-16) # 22*22*100
                AX = np.real(np.fft.ifft2(np.fft.fft2(res,(nx[0],nx[1]),axes=(0,1))*np.conjugate(fft_D[:,:,:,[kk]])))
                midAX = np.sum(AX[:ns[0],:ns[1],:,:],axis=2)
                Smu1 = (midAX*Z+sumD*Sk)*midphi3
                S_mu = np.zeros((ns[0],ns[1],N))
                S_mu[:,:,b[kk,:]==1] = S_sig[:,:,b[kk,:]==1]*Smu1[:,:,b[kk,:]==1]
                Sk = S_mu + S_sig**0.5*npr.randn(ns[0],ns[1],N)
                if Mp[0]==1:
                    maxS = np.max(np.abs(Sk[:]))
                    Sk = Sk/(maxS+1e-16)
                    Sk[abs(Sk)<0.1]=0
                    D[:,:,kk] = D[:,:,kk]*maxS
                
                Sk = Sk*Z
                S[:,:,kk,:]=Sk
                SZk = np.squeeze(b[kk,:].reshape(1,1,N)*Sk)
                fft_SZ[:,:,kk,:] = np.fft.fft2(SZk,(nx[0],nx[1]),axes =(0,1))
                
                """ compute X_{-n} """
                ASZ = np.real(np.fft.ifft2(fft_D[:,:,:,[kk]]*fft_SZ[:,:,[kk],:]))
                res = midres-ASZ
                
                """ Sample Dictionary D """
                SS = np.sum(np.sum(SZk**2,axis=(0,1))*phi)
                D_sig = 1/(SS+D_alpha[:,:,:,kk]+1e-16)
                ASS = np.real(np.fft.ifft2(np.fft.fft2(res,(nx[0],nx[1]),axes=(0,1))*np.conjugate(fft_SZ[:,:,[kk],:])))
                ASS = np.sum(ASS[:nd[0],:nd[1],:,:]*midphi,axis=3)
                ASS = ASS + SS* D[:,:,:,kk]
                D_mu = D_sig*ASS
                D[:,:,:,kk] = D_mu + D_sig**0.5*npr.randn(nd[0],nd[1],Xn)
                fft_D[:,:,:,kk] = np.fft.fft2(D[:,:,:,kk],(nx[0],nx[1]),axes=(0,1))
                
                """ compute X_{-n} """
                ASZ = np.real(np.fft.ifft2(fft_D[:,:,:,[kk]]*fft_SZ[:,:,[kk],:]))
                res = midres-ASZ
            else:
                Sk = npr.randn(ns[0],ns[1],N)/(S_alpha+1e-16)**0.5
                Sk = Sk*Z
                S[:,:,kk,:]=Sk
                fft_SZ[:,:,kk,:] = 0
                D[:,:,:,kk] = npr.randn(nd[0],nd[1],Xn)/(D_alpha[:,:,:,kk]+1e-16)**0.5
                fft_D[:,:,:,kk] = np.fft.fft2(D[:,:,:,kk],(nx[0],nx[1]),axes=(0,1))
                res = midres
            
        """ Sampling pai """
        pia = np.sum(b,1)+pia0
        pib = N-np.sum(b,1)+pib0
        pai = npr.beta(pia,pib)
        
        """ Sampling D_alpha """
        D_alpha = npr.gamma(D_a0+0.5, 1/(D_b0+0.5*D**2+1e-16))
        
        """ Sampling phi """
        c = c0 + 0.5*nXn
        d = d0 + 0.5*np.sum(res**2,axis=(0,1,2))
        phi = npr.gamma(c,1/d)
        
        if iternum%10 == 0:
            mse = np.mean(np.sqrt(np.sum(res**2,axis=(0,1,2))))
            end = time.time()
            print("Iteration %d, mse = %.2f,"
                  " time = %.2fs"
                  % (iternum, mse, end - begin))
            begin = end
    
    result = {"D": D, "S": S, "b": b}
    return result
    

CFA_result = Gibbs(X,nd,K,Mp,nx,N,Xn,L,burnin,num,space,CFA_result)
    

          
       
        
       
        
       
            
    
    
    
    
