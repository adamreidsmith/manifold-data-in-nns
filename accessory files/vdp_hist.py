#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 09:56:48 2019

@author: adamreidsmith
"""

'''
Create datafiles of 1D solution to the Van der Pol equation 
x'' - a(1-x^2)x' + bx = f(t)
'''

import random as r
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import find_peaks

#Parameters
C1 = 1
C2 = np.sqrt(2)
C3 = np.pi/3
T1 = 5                 #w = 1.257
T2 = 10*np.sqrt(2)-2   #w = 0.515
T3 = 30*np.sqrt(3)     #w = 0.121

#RHS
def f(t, phi1, phi2, phi3): 
    return C1*np.cos(2*np.pi/T1*t + phi1) + C2*np.cos(2*np.pi/T2*t + phi2) #+ C3*np.cos(2*np.pi/T3*t + phi3)

data = []

num_ab_pairs = 1
for i in range(num_ab_pairs):
    a = np.random.rand() #Random number in [0,1)
    b = np.random.rand()
    phi1 = 2*np.pi*np.random.rand()
    phi2 = 2*np.pi*np.random.rand()
    phi3 = 2*np.pi*np.random.rand()    
    ic = [2*np.random.rand() - 1, 2*np.random.rand() - 1]

    #Van der Pol oscillator equation
    def vanderpol(ic,t):
        x = ic[0]
        y = ic[1]
        yd = f(t, phi1, phi2, phi3) + a*(1 - x**2)*y - b*x
        xd = y
        return [xd,yd]
    
    #Solve the ivp numerically    
    tmax = 500
    n_points = 10*2**10
    tfull = np.linspace(0,tmax,n_points)
    sol = odeint(vanderpol, ic, tfull)
    
    '''#Plot the data
    plt.figure(figsize=(8,6))
    plt.plot(tfull,sol[:,0])
    plt.xlabel('t')
    plt.ylabel('x(t)')'''
    
    #Keep a random 1000 data points
    #indices = r.sample(range(n_points), k=1000)   
    #indices.sort()
    
    #Keep every tenth data point
    indices = [i for i in range(n_points) if i % 10 == 0]
    t = np.array([tfull[i] for i in indices])
    tmodT1 = t % T1
    x = [sol[i][0] for i in indices]
    xd = [sol[i][1] for i in indices]
    
    n_bins = 100
    
    '''#Plot the data with time modulo T1
    plt.figure(figsize=(8,6))
    plt.scatter(tmodT1, x, s=1)
    plt.xlabel('t mod ' + str(T1))
    plt.ylabel('x(t)')
    plt.title('a = ' + str(a) + ',  b = ' + str(b) + ',  [phi1, phi2] = ' + str([phi1,phi2]))'''
    
    #Plot 2d histogram of t mod T1 vs x
    plt.figure(figsize=(8,6))
    plt.hist2d(tmodT1, x, bins=n_bins)[0]
    plt.xlabel('t mod ' + str(T1))
    plt.ylabel('x(t)')
    plt.title('a = ' + str(a) + ',  b = ' + str(b) + ',  [phi1, phi2] = ' + str([phi1,phi2]))
    
    '''
    #Separate tfull%T1, x, and xdot into sublists of t mod T1
    septime, subtime = [], []
    sepx, subx = [], []
    sepxdot, subxdot = [], []
    for i in range(len(tfull)-1):
        if abs((tfull%T1)[i] - (tfull%T1)[i+1]) < 0.1:
            subtime.append((tfull%T1)[i])
            subx.append(sol[:,0][i])
            subxdot.append(sol[:,1][i])
        else:
            septime.append(subtime)
            sepx.append(subx)
            sepxdot.append(subxdot)
            subtime, subx, subxdot = [], [], []
    
    #3d scatter plot of (t,x,xd)
    fig = plt.figure(figsize=(8,6))
    ax3d = fig.add_subplot(111, projection='3d')
    for i in range(len(septime)):
        ax3d.plot(septime[i], sepx[i], sepxdot[i], c='b')    
    ax3d.set_xlabel(r'$t$')
    ax3d.set_ylabel(r'$x$')
    ax3d.set_zlabel(r'$\dot x$')
    '''                
    
    '''#FFT
    plt.figure(figsize=(8,6))
    fftdata = np.fft.fft(x)
    plt.plot(t,np.abs(fftdata), c='b')
    plt.yscale('log')
    plt.xlabel(r'Frequency')
    plt.ylabel(r'$\hat x (t)$')
    plt.title('Fourier transform of solution to Van der Pol equation')'''
    
    soln = np.array([[t[i],x[i]] for i in range(len(t))])
    fftdata = np.fft.fft(x)
    FT = np.array([[t[i],fftdata[i]] for i in range(len(t))])

    data.append(soln)
    data.append(FT)
    data.append(np.histogram2d(tmodT1, x, bins=n_bins)[0])
    data.append([phi1])
    data.append([a,b]) 
    
    if i % 10 == 0:
        print('Iteration:', i)
                
       
#np.save('../Data_files/vdp_2periods_2000_pts_[soln,FT,hist,[phi1,phi2],[a,b]]', data)
    
    