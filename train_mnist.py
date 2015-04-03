"""
By Zhe Gan, zhe.gan@duke.edu, Duke University, 4.2.2015
"""

import CFA
import numpy as np
import gzip,cPickle

print "Loading data"
f = gzip.open('mnist.pkl.gz', 'rb')
(x_train, t_train), (x_valid, t_valid), (x_test, t_test)  = cPickle.load(f)
f.close()

data = x_train

[N,D] = data.shape
K = 500
Kprime = 500
J = 500
L = 1

learning_rate = 0.01
batch_size = 100
n_iter = 1000
verbose = True

encoder = CFA.CFA(K,Kprime,J,learning_rate,batch_size,n_iter,L,verbose)

print "Iterating"
list_lowerbound = encoder.learning(data)

W1, W2, W3, W4, W5 = encoder.params["W1"], encoder.params["W2"], encoder.params["W3"],\
encoder.params["W4"], encoder.params["W5"]
b1, b2, b3, b4, b5 = encoder.params["b1"], encoder.params["b2"], encoder.params["b3"],\
encoder.params["b4"], encoder.params["b5"]

np.savez("result",ll = list_lowerbound, W1 = W1, W2 = W2, W3 = W3, W4 = W4, 
         W5 = W5, b1 = b1, b2 = b2, b3 = b3, b4 = b4, b5 = b5)
         

import matplotlib.pyplot as plt
plt.style.use('ggplot')

fig = plt.figure()
xvals = np.array(range(1,len(list_lowerbound)+1)) * 5e4
yvals = list_lowerbound
plt.plot(xvals, yvals)
plt.xlabel('training digits used')
plt.ylabel('variational lower bound')
plt.title('MNIST, J = 500, K = 500')
plt.savefig('lowerbound.pdf', bbox_inches = 'tight')
plt.close()