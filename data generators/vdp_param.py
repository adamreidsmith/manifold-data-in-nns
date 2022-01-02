#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:04:35 2019

@author: adamreidsmith
"""

'''
This code solves the Van der Pol equation with forcing term f(t) = cos(wt).
A datafile is created for use in 'nn_param.py'.
'''
import numpy as np
from scipy.integrate import odeint
from os import path, mkdir

###########################################################################
'''
inputs:
    tmax:           The upper bound of the interval [0,tmax] on which to solve 
                        the Van der Pol equation.
                    
    w:              The frequency used in the forcing term.
    
    ic:             The initial condition.
    
    npoints:        The number of time steps to include in each solution.
    
    num_ab_pairs:   The number of times to solve the equation, i.e. the number 
                        of data points.
'''
###########################################################################

def generate_data(tmax=20,
                  w=np.sqrt(2),
                  ic=[0,0],
                  npoints=1000,
                  num_ab_pairs=800):   
     
    #Create a directory to store datafiles if it doesn't aready exist
    if not path.exists('./datafiles'):
        mkdir('./datafiles')
        
    #RHS
    def f(t): 
        return np.cos(w*t)
    
    data = []
    
    for i in range(num_ab_pairs):
        a = np.random.rand() #Random number in [0,1)
        b = np.random.rand() #Random number in [0,1)
        
        #Van der Pol oscillator equation
        def vanderpol(ic,t):
            x = ic[0]
            y = ic[1]
            yd = f(t) + a*(1-x**2)*y - b*x
            xd = y
            return [xd,yd]
        
        #Solve the ivp numerically    
        t = np.linspace(0,tmax,npoints)
        sol = odeint(vanderpol, ic, t)
        
        x = sol[:,0]
        
        t = np.insert(t,0,a)
        x = np.insert(x,0,b)
        soln = [[t[j],x[j]] for j in range(len(t))]
        data.append(soln)
            
        if i % 10 == 0 and __name__ == '__main__':
            print('Iteration:', i)
    
    np.save('./datafiles/vdp_param_data', data)
    
        
if __name__ == '__main__':
    generate_data()
        