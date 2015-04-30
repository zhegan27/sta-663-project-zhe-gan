
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
import scipy.io
import cfa


np.random.seed(1234)

""" loading data. """
data = scipy.io.loadmat('mnist_100.mat')
N, Nx, Ny, X = data['N'], data['Nx'], data['Ny'], data['X']

N = np.array(N)[0]
Nx = np.array(Nx)[0]
Ny = np.array(Ny)[0]

""" experimental setup. """
X = X.reshape(Nx,Ny,1,N,order='F')
nd = np.array([7,7]) # layer one dictionary size
Mp = np.array([3,3]) # max-pooling ratio is set to 3.
K = 36 # the number of factors
Xn = 1
L = 1 # index of layer
nx = np.array([28,28])

burnin = 200
num = 50
space = 6

CFA_result, rmse, rX = cfa.Gibbs(X,nd,K,Mp,nx,N,Xn,L,burnin,num,space)
    
""" plot mse. """
cfa.plotMSE(rmse,'mnist_mse')

""" plot reconstruction. """
cfa.DispDic(data['X'],'mnist_original')
cfa.DispDic(rX,'mnist_reconstruction')

""" visualize the dictionaries. """
Dic = CFA_result["D"].reshape(nd[0],nd[1],K,order='F')
cfa.DispDic(Dic,'mnist_dictionary')

       
        
       
        
       
            
    
    
    
    
