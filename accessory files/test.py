#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 16:06:55 2019

@author: adamreidsmith
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import so3_data_generation
import scipy

'''
def wfm(X, w):
    M = [X[0]]
    for i in range(1,len(X)):
        M.append(twosphere_geodesic(M[-1], X[i], w[i]/sum(w[:i+1])))
    return np.array(M)

def wfm2(X, w):
    M = X[0].unsqueeze(0)
    for i in range(1,len(X)):
        M = torch.cat( (M, twosphere_geodesic2(M[-1], X[i], torch.div(w[i],sum(w[:i+1]))).unsqueeze(0)) )
    return M


def twosphere_geodesic(x1,x2,t):
    x1 = np.array(x1/np.linalg.norm(x1))
    x2 = np.array(x2/np.linalg.norm(x2))
    
    y1 = x2 - np.dot(x1,x2)*x1
    y2 = y1/np.linalg.norm(y1)
    
    c = np.arccos(np.dot(x1,x2))
    
    return np.cos(t*c)*x1 + np.sin(t*c)*y2

def twosphere_geodesic2(x1,x2,t):
    x1 = torch.div(x1,torch.norm(x1))
    x2 = torch.div(x2,torch.norm(x2))
    
    y1 = x2 - torch.dot(x1,x2)*x1
    y2 = y1/torch.norm(y1)
    
    c = torch.acos(torch.dot(x1,x2))
    
    return torch.cos(t*c)*x1 + torch.sin(t*c)*y2


X = []
for _ in range(1000):
    x = np.array([2*np.random.rand()-1,2*np.random.rand()-1,2*np.random.rand()-1])
    x = x/np.linalg.norm(x)
    X.append(list(x))

X = np.array(X)

fig = plt.figure(figsize=(8,6))
ax3d = fig.add_subplot(111, projection='3d')
ax3d.scatter(X[:,0], X[:,1], X[:,2], c='b')

WFM = wfm(X, np.ones(len(X)))

ax3d.scatter(WFM[:,0], WFM[:,1], WFM[:,2], c='r')'''

def power(A,k):
    #Computes matrix power A^k where k is a non-negative integer
    #A can be a square matrix or a batch of square matrices
    if k == 0 and len(A.shape) == 2:
        return torch.eye(A.shape[-1])
    
    if k == 0 and len(A.shape) == 3:
        inter = torch.eye(A.shape[-1]).unsqueeze(0)
        return inter.repeat(A.shape[0],1,1)
    
    if k == 1:
        return A
    
    P = torch.matmul(A,A)
    for i in range(k-2):
        P = torch.matmul(P,A)
    return P
        
    
def expm(A, tol=1e-8, maxit=20):
    #Matrix exponential using power series definition
    #A can be a square matrix or a batch of square matrices
    shape = A.shape
    batched = False if len(shape) == 2 else True
    eps = 10.0
    exp = power(A,0)
    k = 1
    while eps > tol and k < maxit:
        old_exp = exp.clone()
        exp += torch.mul(1/np.math.factorial(k), power(A,k))
        k += 1
        
        if batched:
            inter1 = exp - old_exp
            inter2 = torch.mul(inter1, inter1).view(shape[0], shape[1]*shape[2])
            eps = torch.max(torch.sqrt(inter2).sum(dim=1))
        else:
            eps = torch.norm(exp - old_exp)
        
    if k >= maxit:
        print('Maximum iterations in \'expm\' reached.', 
              'Tolerance of %e achieved in %d iterations.' % (eps, k))
        
    return exp

def trace(A):
    shape = A.shape
    if len(shape) == 2:
        return torch.trace(A)
    mask = torch.eye(shape[-1]).unsqueeze(0).repeat(shape[0],1,1)
    return torch.mul(mask, A).view(shape[0], shape[1]*shape[2]).sum(dim=1)


def so3_geodesic(A1,A2,t):
    P = torch.matmul(torch.inverse(A1), A2)
    
    shape = P.shape
    
    theta = torch.acos(torch.div(trace(P) - 1, 2))
    
    C = torch.div(theta, torch.mul(2, torch.sin(theta)))

    if len(shape) == 2:
        G = torch.mul(C, P - P.transpose(0,1))
    else:
        inter = P - P.transpose(1,2)
        G = torch.mul(C.unsqueeze(1), inter.view(shape[0], shape[1]*shape[2]))
        G = G.view(shape)
        
    O = torch.matmul(A1, expm(torch.mul(t, G)))
    
    return O

data = so3_data_generation.generate_data([1,2,2.9], 2, 5, 10e-4)
A1 = torch.Tensor(data[0])
A1.requires_grad = True
A2 = torch.Tensor(data[1])
A2.requires_grad = True

import timeit
s = """\
import torch
def trace(A):
    shape = A.shape
    if len(shape) == 2:
        return torch.trace(A)
    mask = torch.eye(shape[-1]).unsqueeze(0).repeat(shape[0],1,1)
    return torch.mul(mask, A).view(shape[0], shape[1]*shape[2]).sum(dim=1)
L = torch.randn(5,3,3)
"""



print(timeit.timeit(stmt='torch.abs(torch.acos((trace(L) - 1)/2))', setup=s, number=10000))

print(timeit.timeit(stmt='trace(torch.matmul(L.transpose(1,2), L))', setup=s, number=10000))









