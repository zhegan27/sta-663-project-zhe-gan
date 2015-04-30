
"""
Reconstruction using PCA

@author: Zhe Gan (zhe.gan@duke.edu), Duke ECE, 4.30.2015
"""

import numpy as np
import numpy.linalg as npl
import scipy.io

""" loading mnist data. """
data = scipy.io.loadmat('mnist_100.mat')
N, Nx, Ny, X = data['N'], data['Nx'], data['Ny'], data['X']

N = np.array(N)[0]
X = X.reshape(28*28,N,order='F')

u, s, v = npl.svd(X)

K = 36
Xrec = np.dot(np.dot(u[:,:K],np.diag(s[:K])),v[:K,:])

res = X - Xrec
mnist_mse = np.mean(np.sqrt(np.sum(res**2,axis=0)))

""" loading face data. """
data = scipy.io.loadmat('face101_64_128.mat')
trainX = data['trainX']
testX = data['testX']
L1X = np.concatenate((trainX,testX),1)

size_L1X = L1X.shape
Nx = int(np.sqrt(size_L1X[0])) # Width of image
Ny = Nx # Height of image
N = size_L1X[1] # Number of image

# centering the density of pixel
X = L1X - np.mean(L1X)

u, s, v = npl.svd(X)

K = 36
Xrec = np.dot(np.dot(u[:,:K],np.diag(s[:K])),v[:K,:])

res = X - Xrec
face_mse = np.mean(np.sqrt(np.sum(res**2,axis=0)))

