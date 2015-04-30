
"""
Bayesian Posterior Inference for the Convolutional Factor Analysis 
via Gibbs sampling

@author: Zhe Gan (zhe.gan@duke.edu), Duke ECE, 4.30.2015

Reference:
    1. B. Chen, G. Polatkan, G. Sapiro, D. Dunson and L. Carin, "
    The Hierarchical Beta Process for Convolutional Factor Analysis and 
    Deep Learning", Proc. Int. Conf. Machine Learning (ICML), 2011

    2. B. Chen, G. Polatkan, G. Sapiro, D. Blei, D. Dunson and L. Carin, 
    "Deep Learning with Hierarchical Convolutional Factor Analysis", IEEE 
    Trans. Pattern Analysis & Machine Intelligence, 2013.
"""

import numpy as np
import numpy.random as npr
import numpy.fft as npf
import time
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def Gibbs(X,nd,K,Mp,nx,N,Xn,L,burnin,num,space):
    """ Gibbs sampling algorithm for the CFA model. 
    Input:
        X:   input data nx * ny * N
        nd:  dictionary size
        K:   number of factors
        Mp:  max-pooling ratio
        nx:  input data size
        N:   number of datapoints
        Xn:  number of input channels
        L:   index of layers
        burnin,num,space:  Gibbs sampling parameters
        CFA_result: results of the previous layer
    """

    nx2 = nx[0] * nx[1]
    ns = nx - nd + 1
    nXn = nx[0] * nx[1] * Xn
    
    """ set hyper parameters. """
    c0,d0,g0,h0,D_a0,D_b0 = 1e-6,1e-6,1e0,1e-3,1e0,1e-3
    
    """ initialization. """
    D = np.zeros((nd[0],nd[1],Xn,K))
    D_alpha = npr.gamma(D_a0+0.5, 1/(D_b0+0.5*D**2))
    phi = nx2*np.ones(N)
    a = 1.0
    pia0 = a/K
    pib0 = 1.0-pia0
    pai = npr.beta(pia0,pib0*np.ones(K))
    b = np.ones((K,N))
    P = np.zeros((Mp))
    P[0,0] = 1.0
    Z = np.ones(((nx-nd)/Mp+1))
    Z = np.tile(Z,N)
    Z = np.kron(Z,P).reshape(ns[0]+Mp[0]-1,ns[1]+Mp[1]-1,N,order='F')
    Z = Z[:-1-Mp[0]+2,:-1-Mp[1]+2,:]
    Ind = (Z==1.0).nonzero()
    S_alpha = np.zeros((ns[0],ns[1],N))
    S = npr.randn(ns[0],ns[1],K,N)
    S = Z.reshape(ns[0],ns[1],1,N,order='F')*S
    bSZ = b.reshape(1,1,K,N,order='F')*S
    
    fft_D = npf.fft2(D,(nx[0],nx[1]),axes = (0,1))
    fft_SZ = npf.fft2(bSZ,(nx[0],nx[1]),axes = (0,1))
    res = X;
    iternum = 0; 

    """ collect results. """
    rD = np.zeros((nd[0],nd[1],Xn,K))    
    rS = np.zeros((ns[0],ns[1],K,N))
    rb = np.zeros((K,N))
    rmse = np.array([])
    
    """ Gibbs sampling. """
    begin = time.time()
    while (iternum < burnin+num*space):
        iternum = iternum + 1
        
        midphi = phi.reshape(1,1,1,N,order='F')
        midphi3 = phi.reshape(1,1,N,order='F')
        
        for kk in range(K):
            
            """ 1. compute X_{-n} """            
            ASZ = np.real(npf.ifft2(fft_D[:,:,:,[kk]]*fft_SZ[:,:,[kk],:],axes=(0,1)))
            midres = res + ASZ
            
            """ 2. Sample b """
            Sk = S[:,:,kk,:]
            P = -(res**2 - midres**2)*midphi/2.0
            P = np.sum(P,axis=(0,1,2)) + (np.log(pai[kk])-np.log(1-pai[kk]))
            b[kk,:] = npr.rand(N) <np.exp(P)
            fft_SZ[:,:,kk,:] = npf.fft2(b[kk,:].reshape(1,1,N,order='F')*Sk,(nx[0],nx[1]),axes=(0,1))
            ASZ = np.real(npf.ifft2(fft_D[:,:,:,[kk]]*fft_SZ[:,:,[kk],:],axes=(0,1)))
            res = midres - ASZ
            
            """ 3. Sample S """
            sumD = np.sum(D[:,:,:,kk]**2)
            S_alpha[Ind] = npr.gamma(g0 + 0.5,1./(h0+ 0.5*(Sk[Ind]**2)))
            
            if np.sum(b[kk,:]) > 0:
                S_sig = 1.0/(sumD*midphi3*Z*b[kk,:].reshape(1,1,N,order='F') + S_alpha + 1e-16) 
                AX = np.real(npf.ifft2(npf.fft2(res,(nx[0],nx[1]),axes=(0,1))*np.conjugate(fft_D[:,:,:,[kk]]),axes=(0,1)))
                midAX = np.sum(AX[:ns[0],:ns[1],:,:],axis=2)
                Smu1 = (midAX*Z+sumD*Sk)*midphi3
                S_mu = np.zeros((ns[0],ns[1],N))
                S_mu[:,:,b[kk,:]==1.0] = S_sig[:,:,b[kk,:]==1.0]*Smu1[:,:,b[kk,:]==1.0]
                Sk = S_mu + S_sig**0.5*npr.randn(ns[0],ns[1],N)
                if Mp[0]==1:
                    maxS = np.max(np.abs(Sk[:]))
                    Sk = Sk/(maxS+1e-16)
                    Sk[abs(Sk)<0.1]=0
                    D[:,:,kk] = D[:,:,kk]*maxS
                
                Sk = Sk*Z
                S[:,:,kk,:]=Sk
                SZk = (b[kk,:].reshape(1,1,N,order='F')*Sk)
                fft_SZ[:,:,kk,:] = npf.fft2(SZk,(nx[0],nx[1]),axes =(0,1))
                
                """ 4. compute X_{-n} """
                ASZ = np.real(npf.ifft2(fft_D[:,:,:,[kk]]*fft_SZ[:,:,[kk],:],axes=(0,1)))
                res = midres - ASZ
                
                """ 5. Sample Dictionary D """
                SS = np.sum(np.sum(SZk**2,axis=(0,1))*phi)
                D_sig = 1.0/(SS+D_alpha[:,:,:,kk]+1e-16)
                ASS = np.real(npf.ifft2(npf.fft2(res,(nx[0],nx[1]),axes=(0,1))*np.conjugate(fft_SZ[:,:,[kk],:]),axes=(0,1)))
                ASS = np.sum(ASS[:nd[0],:nd[1],:,:]*midphi,axis=3)
                ASS = ASS + SS* D[:,:,:,kk]
                D_mu = D_sig*ASS
                D[:,:,:,kk] = D_mu + D_sig**0.5*npr.randn(nd[0],nd[1],Xn)
                fft_D[:,:,:,kk] = npf.fft2(D[:,:,:,kk],(nx[0],nx[1]),axes=(0,1))
                
                """ 6. compute X_{-n} """
                ASZ = np.real(npf.ifft2(fft_D[:,:,:,[kk]]*fft_SZ[:,:,[kk],:],axes=(0,1)))
                res = midres - ASZ
            else:
                Sk = npr.randn(ns[0],ns[1],N)/(S_alpha+1e-16)**0.5
                Sk = Sk*Z
                S[:,:,kk,:]=Sk
                fft_SZ[:,:,kk,:] = 0.0
                D[:,:,:,kk] = npr.randn(nd[0],nd[1],Xn)/((D_alpha[:,:,:,kk]+1e-16)**0.5)
                fft_D[:,:,:,kk] = npf.fft2(D[:,:,:,kk],(nx[0],nx[1]),axes=(0,1))
                res = midres
            
        """ 7. Sampling pai """
        pia = np.sum(b,1)+pia0
        pib = N-np.sum(b,1)+pib0
        pai = npr.beta(pia,pib)
        
        """ 8. Sampling D_alpha """
        D_alpha = npr.gamma(D_a0+0.5, 1.0/(D_b0+0.5*D**2+1e-16))
        
        """ 9. Sampling phi """
        c = c0 + 0.5*nXn
        d = d0 + 0.5*np.sum(res**2,axis=(0,1,2))
        phi = npr.gamma(c,1.0/d)
        
        """ 10. Save samples. """
        if iternum > burnin and iternum%space == 0:
            rD = rD + D/num
            rS = rS + S/num
            rb = rb + b/num
        
        if iternum%1 == 0:
            mse = np.mean(np.sqrt(np.sum(res**2,axis=(0,1,2))))
            end = time.time()
            print("Iteration %d, mse = %.2f, time = %.2fs"
                  % (iternum, mse, end - begin))
            begin = end
            rmse = np.append(rmse,mse)
    
    CFA_result = {"D": rD, "S": rS, "b": rb}
    
    """ obtain reconstruction result. """
    fft_D = npf.fft2(rD,(nx[0],nx[1]),axes=(0,1)).reshape(nx[0],nx[1],Xn,K,1,order='F')
    rbSZ = rS * rb.reshape(1,1,K,N,order='F')
    fft_SZ = npf.fft2(rbSZ,(nx[0],nx[1]),axes=(0,1)).reshape(nx[0],nx[1],1,K,N,order='F')
    ASZ = np.real(npf.ifft2(fft_D * fft_SZ,axes=(0,1)))
    rX = np.squeeze(np.sum(ASZ,axis = 3))
    
    return CFA_result, rmse, rX
    

def plotMSE(rmse,name):

    plt.figure()
    xvals = np.array(range(1,len(rmse)+1))
    yvals = rmse
    plt.plot(xvals, yvals)
    plt.xlabel('iteration number')
    plt.ylabel('RMSE')
#    plt.title('MNIST')
    plt.savefig(name + '.pdf', bbox_inches = 'tight')
    plt.close()
    
def DispDic(D,name):
    
    """ Display the dictionary elements as an image. """
    
    Nx, Ny, N = D.shape
    RowNum = np.floor(N**0.5).astype('int')
    ColNum = np.ceil(N/RowNum).astype('int')
    
    plt.figure()
    I = np.zeros((RowNum*(Nx+1), ColNum*(Ny+1),3))
    I[:,:,2] = 1
    for row in range(RowNum):
        for col in range(ColNum):
            n = row*ColNum + col
            if n < N:
                D[:,:,n] = D[:,:,n] - np.min(D[:,:,n])
                MaxD = np.max(D[:,:,n])
                if MaxD > 0:
                    D[:,:,n] = D[:,:,n]/MaxD   
                I[ row*(Nx+1)+1 : (row+1)*(Nx+1) , col*(Ny+1)+1 : (col+1)*(Ny+1),0] = D[:,:,n]
                I[ row*(Nx+1)+1 : (row+1)*(Nx+1) , col*(Ny+1)+1 : (col+1)*(Ny+1),1] = D[:,:,n]
                I[ row*(Nx+1)+1 : (row+1)*(Nx+1) , col*(Ny+1)+1 : (col+1)*(Ny+1),2] = D[:,:,n]
            else:
                I[ row*(Nx+1)+1 : (row+1)*(Nx+1) , col*(Ny+1)+1 : (col+1)*(Ny+1),0] = 0
                I[ row*(Nx+1)+1 : (row+1)*(Nx+1) , col*(Ny+1)+1 : (col+1)*(Ny+1),1] = 0
                I[ row*(Nx+1)+1 : (row+1)*(Nx+1) , col*(Ny+1)+1 : (col+1)*(Ny+1),2] = 0
                
    plt.imshow(I) 
    plt.axis('off')
    plt.savefig(name + '.pdf', bbox_inches = 'tight')
    plt.close()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    