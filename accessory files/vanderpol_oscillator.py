#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:57:42 2019

@author: adamreidsmith
"""

import math as m
import random as r
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#Parameters
a = .3
b = .8
C1 = 1
#C2 = np.pi/2
T1 = 5
#T2 = 100*np.sqrt(2)

#RHS
def f(t): 
    return C1*np.cos(2*np.pi/T1*t) #+ C2*np.cos(2*np.pi/T2*t)

#Van der Pol oscillator equation
def vanderpol(ic,t):
    x = ic[0]
    y = ic[1]
    yd = f(t) + a*(1-x**2)*y - b*x
    xd = y
    return xd, yd

#Solve the ivp numerically with many points to ensure accuracy
ic = [0,0]
t_max = 500
n_points = 50000
t = np.linspace(0,t_max,n_points)
sol = odeint(vanderpol, ic, t)

#Plot the data
plt.figure(figsize=(8,6))
plt.plot(t,sol[:,0])
#plt.plot(t,xd)
plt.xlabel('t')
plt.ylabel('x(t)')

#Keep a random 1/50 of the data (1000 points)
indices = r.choices(range(n_points), k=1000)
indices.sort()
t = np.array([t[i] for i in indices])
tmodT1 = t%T1
x = [sol[i][0] for i in indices]
xd = [sol[i][1] for i in indices]

'''
plt.figure(2, figsize=(8,6))
plt.plot(x, xd)
plt.xlabel('x')
plt.ylabel('dxdt')
'''


#Plot the data
plt.figure(figsize=(8,6))
plt.scatter(tmodT1,x,s=1)
plt.xlabel('t mod T')
plt.ylabel('x(t)')

bin_edges = np.linspace(0,T1, 100)
n_bins = len(bin_edges)
binwidth = bin_edges[1] - bin_edges[0]

bins = []

for i in range(n_bins):
    bin_total = 0
    for s in tmodT1:
        if bin_edges[i] <= s < bin_edges[i] + binwidth:
            bin_total += x[i]
    bins.append(bin_total)


plt.figure(figsize=(8,6))
plt.scatter(bin_edges, bins)


#Finite difference cefficients for calculating numerical derivative
#Stencil is a list of integers [-n, -n+1, ..., n-1, n]
#Coefficients is a list of reals [c_1, ..., c_2n+1]
#d^sf(x0)/dx^s =  (1/h^s) * ( c_1*f(x0 - n*h) + c_2*f(x0 - (n-1)*h) + ... + c_2n+1*f(x0 + n*h) )
def f_diff_coeff(len_stencil, deg):
    stencil = np.array([-np.floor(len_stencil/2) + i for i in range(len_stencil)])
    stencil_matrix = np.array([ [stencil[i]**n for i in range(len(stencil))] for n in range(len(stencil)) ])
    deg_vec = [ [m.factorial(deg)] if deg == i else [0] for i in range(len_stencil) ]
    return np.dot(np.linalg.inv(stencil_matrix),deg_vec).T[0], stencil
    
h = t[1] - t[0]
deg = 1
len_stencil = 20

#Return the derivative of a list of function values representing a function
def get_deriv(fun, index, deg, len_stencil, h):
    coeffs, stencil = f_diff_coeff(len_stencil, deg)
    if index <= len(fun) - np.ceil(len_stencil/2) and index >= np.ceil(len_stencil/2):
        return (1/h**deg) * sum([coeffs[i]*fun[int(index + stencil[i])] for i in range(len_stencil)])
    return 0

#Plot the derivative of x to compare
xp = [get_deriv(x, index, deg, len_stencil, h) for index in range(len(x))]
plt.figure(1)
plt.plot(t,xp)

#Calculate a and b from only x, t, and the form of the ode
i1=100; i2=900 #arbitrary distinct indices not within len_stencil/2 of the ends of the list

x1 = x[i1] #x
xd1 = get_deriv(x, i1, 1, len_stencil, h) #x dot
xdd1 = get_deriv(x, i1, 2, len_stencil, h) #x double dot
x2 = x[i2]
xd2 = get_deriv(x, i2, 1, len_stencil, h)
xdd2 = get_deriv(x, i2, 2, len_stencil, h)

ac,bc = np.dot( np.linalg.inv([ [(x1**2 - 1)*xd1, x1],
                                [(x2**2 - 1)*xd2, x2]]), [ [f(t[i1]) - xdd1],
                                                           [f(t[i2]) - xdd2] ]).T[0]
print(ac, bc)


#Add normal noise to the data
def add_noise(data,sigma):
    for i in range(len(data)):
        for j in range(len(data[0])):
            data[i][j] += np.random.normal(0,sigma) 


data = [[t[i], x[i]] for i in range(len(t))]
#np.savetxt('../Data_files/vdp_data.csv', data)

'''
fft = np.fft.fft(x)

plt.figure(1)
plt.plot(t,fft)
plt.ylim(-3,3)
'''
