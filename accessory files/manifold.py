#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 10:41:51 2019

@author: adamreidsmith
"""

import random
import numpy as np
import matplotlib.pyplot as plt

def manifold(data, tol, n_points):
    
    #Number of samples
    N = len(data)
    
    #Number of manifold points
    K = n_points
    
    #Initialize manifold points and parameters
    g = []; P = []; Pi = []; F = [0]
    
    g.append( [data[i] for i in random.sample(range(N), k=K)] )
    P.append( [1/K for _ in range(K)] )
    l = 8
    
    def normsq(x):
        return sum([y**2 for y in x])
    
    def iterate(g,P,n):
        
        def pi(y):
            return sum([P[n][k]*np.exp((-1/l)*normsq(y-g[n][k])) for k in range(K)])
        Pi.append( pi )
                
        def f(y,k):
            return (P[n][k]/Pi[n](y))*np.exp((-1/l)*normsq(y-g[n][k]))
        F.append( f )
                
        #Update P
        P.append([k for k in range(K)]) #Lengthen P
        for k in range(K):
            P[n+1][k] = np.mean([F[n+1](data[i],k) for i in range(N)])

        #Update g
        g.append([np.array([0. for _ in range(len(g[0][0]))]) for _ in range(K)]) #Lengthen g
        for a in range(len(g[0][0])):
            for k in range(K):
                g[n+1][k][a] = (1/(N*P[n+1][k]))*sum([(data[i][a])*F[n+1](data[i],k) for i in range(N)])
    
    error = tol + 1
    it = 0;
    while error > tol:
        iterate(g,P,it)
        it += 1
        error = max([normsq(g[it][k] - g[it-1][k]) for k in range(K)])
        print('It:', it, ' Error:', error)
    
    return g


n_data = 2000
radius = 20
data = np.array([(np.random.normal(0,1)+radius*np.cos(t), 
                  np.random.normal(0,1)+radius*np.sin(t)) 
                  for _ in range(n_data) for t in [random.random()*np.pi]])

plt.figure(1)
plt.scatter(data[:,0],data[:,1], s=1)

g = manifold(data,0.1,75)

x = [g[-1][i][0] for i in range(len(g[-1]))]
y = [g[-1][i][1] for i in range(len(g[-1]))]

plt.scatter(x,y,s=5,c='r')

'''

data = np.loadtxt('vdp_data.csv')

g = manifold(data,0.1,100)
    
t = [g[-1][i][0] for i in range(len(g[-1]))]
x = [g[-1][i][1] for i in range(len(g[-1]))]

plt.figure(1)
plt.scatter(data[:,0],data[:,1], c='b', s=1)
plt.scatter(t, x, s=5, c='r')
'''