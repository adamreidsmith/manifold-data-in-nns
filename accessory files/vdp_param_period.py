#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 13:41:54 2019

@author: adamreidsmith
"""

import random as r
import numpy as np
from scipy.integrate import odeint
#import matplotlib.pyplot as plt

#Parameters
C1 = 1
#C2 = np.pi/2
T1 = 5
#T2 = 100*np.sqrt(2)

#RHS
def f(t): 
    return C1*np.cos(2*np.pi/T1*t)  # + C2*np.cos(2*np.pi/T2*t)

data = []
datanotmod = []

num_ab_pairs = 1000
for i in range(num_ab_pairs):
    a = np.random.rand() #Random number in [0,1)
    b = np.random.rand()
    
    #Van der Pol oscillator equation
    def vanderpol(ic,t):
        x = ic[0]
        y = ic[1]
        yd = f(t) + a*(1-x**2)*y - b*x
        xd = y
        return [xd,yd]
    
    #Solve the ivp numerically    
    ic = [0,0]
    tmax = 80
    n_points = 10000
    t = np.linspace(0,tmax,n_points)
    sol = odeint(vanderpol, ic, t)
    
    '''
    #Plot the data
    plt.figure(figsize=(8,6))
    plt.plot(t,sol[:,0])
    plt.xlabel('t')
    plt.ylabel('x(t)')
    '''
    
    #Keep a random 1000 data points
    indices = r.sample(range(n_points), k=4000)
    indices.sort()
    t = np.array([t[i] for i in indices])
    tmodT1 = t % T1
    x = [sol[i][0] for i in indices]
    xd = [sol[i][1] for i in indices]
    
    '''
    #Plot the data with time modulo T1
    plt.figure(figsize=(8,6))
    plt.scatter(tmodT1, x, s=1)
    '''
    
    tmodT1 = np.insert(tmodT1,0,a)
    x = np.insert(x,0,b)
    soln = [[tmodT1[j],x[j]] for j in range(len(tmodT1))]
    data.append(soln)
    
    t = np.insert(t,0,a)
    soln2 = [[t[j],x[j]] for j in range(len(t))]
    datanotmod.append(soln2)
    
    if i % 10 == 0:
        print('Iteration:', i)

np.save('../Data_files/vdp_param_data_modT1', data)
np.save('../Data_files/vdp_param_data_modT1_notmod', datanotmod)
    
    