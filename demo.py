# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 12:06:36 2015

@author: Zhe Gan
"""
import CFA
import numpy as np
import scipy.io

mat = scipy.io.loadmat('mnist_part.mat')
N,Nx,Ny,X = mat['N'],mat['Nx'],mat['Ny'],mat['X']


X = X.reshape(Nx,Ny,1,N)
nd = np.array([7,7])
Mp = np.array([3,3])
K = 20
nx = np.array([28,28])
N = np.array(N)[0]
        

burnin = 200
num = 50
space = 6
CFA_result = []
[CFA_result, inputdata]=CFA.CFA_easy(X,nd,K,Mp,nx,N,burnin,num,space,CFA_result)
