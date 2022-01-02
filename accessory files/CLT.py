#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 13:12:31 2019

@author: adamreidsmith
"""
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

n = 100 #Length of a sample
N = 10000 #Number of samples

#F distribution parameters
d1 = 12.
d2 = 128.

data = np.loadtxt('../Data_files/vdp_data.csv')

mu = d2/(d2-2.)
sigma = np.sqrt( (2.*d2**2.*(d1+d2-2.)) / (d1*(d2-2.)**2.*(d2-4.)) )

#Gaussian function with mean mu and standard deviation sigma
def Gaussian(x,mu,sigma):
    return 1./np.sqrt(2. * np.pi * sigma**2) * np.exp(-(x - mu)**2. / (2. * sigma**2.))

#Sn = [np.mean(np.random.normal(mu, sigma, n)) for _ in range(N)]  #Sample means from a normal distribution.
Sn = [np.mean(np.random.f(d1, d2, n)) for _ in range(N)]  #Sample means from an F-distribution.
#CLT says distribution can be arbitrary as long as it has mean mu and variance sigma**2.
Vn = [np.sqrt(n) * (Sn[i] - mu) for i in range(N)]

#Plot normalized histogram
plt.figure(1,figsize=(8,6))
plt.hist(Vn, normed=True, bins=50)

x = np.linspace(min(Vn), max(Vn), 200)

#Fit a Gaussian to the histogram
(mu_fit, sigma_fit) = norm.fit(Vn)
y_fit = Gaussian(x, mu_fit, sigma_fit)
plt.plot(x,y_fit,'g')

#Plot Gaussian with mean 0 and stdev sigma
y = Gaussian(x, 0, sigma)
plt.plot(x,y,'r')

plt.legend(('Gaussian fit to histogram',r'$\mathcal{N}(\mu,\sigma)$'))
plt.title('Normalized histogram of sample means')





