
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
data = scipy.io.loadmat('face101_64_128.mat')
trainX = data['trainX']
testX = data['testX']
L1X = np.concatenate((trainX,testX),1)

size_L1X = L1X.shape
Nx = int(np.sqrt(size_L1X[0])) # Width of image
Ny = Nx # Height of image
N = size_L1X[1] # Number of image

# centering the density of pixel
X = (L1X - np.mean(L1X)).reshape((Nx, Ny, 1, N), order='F')

""" experimental setup. """
nd = np.array([7,7]) # convolutional dictionary size
Mp = np.array([3,3]) # image patch size
K = 36 # number of convolutional dictionaries at each layer
Xn = 1 # number of at the bottom layer
L = 1 # number of layers
nx = np.array([Nx,Ny]) # image size

burnin = 200
num = 50
space = 6

CFA_result, rmse, rX = cfa.Gibbs(X,nd,K,Mp,nx,N,Xn,L,burnin,num,space)
    
""" plot mse. """
cfa.plotMSE(rmse,'face_mse')

""" plot reconstruction. """
cfa.DispDic(L1X.reshape(Nx,Ny,N,order='F'),'face_original')
cfa.DispDic(rX,'face_reconstruction')

""" visualize the dictionaries. """
Dic = CFA_result["D"].reshape(nd[0],nd[1],K,order='F')
cfa.DispDic(Dic,'face_dictionary')

       
        
       
        
       
            
    
    
    
    
