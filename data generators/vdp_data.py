#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 09:56:48 2019

@author: adamreidsmith
"""

'''
Create datafiles of 1D solution to the Van der Pol equation: x'' - a*(1 - x^2)*x' + b*y = f(t).

Datafiles include the computed solution, its fast Fourier transform, a histogram
of x(t) vs t mod T where T is the first period of f, the phase(s) of f, and the 
parameters a and b.

These datafiles are created for use in 'nn_hist.py', 'nn_ft.py', and 'nn_wavelet.py'. 
'''

import numpy as np
from scipy.integrate import odeint
from os import path, mkdir

###############################################################################
'''
Inputs:
    tmax:           The upper bound of the interval [0,tmax] on which to solve 
                        the Van der Pol equation.
    
    initial_cond:   Initial condition. Can be set to 'random' or a list of length 2.
    
    n_points:       The number of time steps to include in each solution.
    
    num_ab_pairs:   The number of times to solve the equation, i.e. the number 
                        of data points.
    
    n_periods:      Number of periodic terms in the forcing function.
    
    include_phase:  Include or exclude a random phase in the forcing terms.
    
    C:              Coefficients of the terms in the forcing function. Must be 
                        a list of length 'n_periods'.
    
    T:              Periods of the terms in the forcing function. Must be a list 
                        of length 'n_periods'.
    
    file_name:      Name of the datafile.
'''
###############################################################################

def generate_data(tmax=500,
                  initial_cond='random',
                  n_points=2**10,
                  num_ab_pairs=800,
                  n_periods=3,
                  include_phase=True,
                  C=[1, np.sqrt(2), np.pi/3],
                  T=[5, 10*np.sqrt(2)-2, 30*np.sqrt(3)],
                  file_name=None):
    
    twopi = 2*np.pi
    
    #Create a directory to store datafiles if it doesn't aready exist
    if not path.exists('./datafiles'):
        mkdir('./datafiles')
    
    assert type(C) == type(T) == list and n_periods == len(C) == len(T), \
           'C and T must be lists of length \'n_periods\'.'

    #RHS
    def f(t, phi, C, T):
        val = 0
        if include_phase:
            for i in range(n_periods):
                val += C[i] * np.cos(twopi/T[i]*t + phi[i])
            return val
        else:
            for i in range(n_periods):
                val += C[i] * np.cos(twopi/T[i]*t)
            return val
        
    data = []
    
    for i in range(num_ab_pairs):
        a = np.random.rand() #Random number in [0,1)
        b = np.random.rand() #Random number in [0,1)
        
        if initial_cond == 'random':
            ic = [2*np.random.rand() - 1, 2*np.random.rand() - 1]
        else:
            ic = initial_cond
        
        phi = []
        if include_phase:
            for _ in range(n_periods):
                phi.append(twopi*np.random.rand())
    
        #Van der Pol oscillator equation
        def vanderpol(ic,t):
            x = ic[0]
            y = ic[1]
            yd = f(t, phi, C, T) + a*(1 - x**2)*y - b*x
            xd = y
            return [xd,yd]
        
        #Solve the ivp numerically    
        npoints = 10*n_points
        tfull = np.linspace(0,tmax,npoints)
        sol = odeint(vanderpol, ic, tfull)
        
        #Keep every tenth data point
        indices = [i for i in range(npoints) if i % 10 == 0]
        t = np.array([tfull[i] for i in indices])
        tmodT1 = t % T[0]
        x = [sol[i][0] for i in indices]
        
        n_bins = 100
        
        soln = np.array([[t[i],x[i]] for i in range(len(t))])
        fftdata = np.fft.fft(x)
        FT = np.array([[t[i],fftdata[i]] for i in range(len(t))])
    
        data.append(soln)
        data.append(FT)
        data.append(np.histogram2d(tmodT1, x, bins=n_bins)[0])
        data.append(phi)
        data.append([a,b]) 
        
        if i % 10 == 0 and __name__ == '__main__':
            print('Iteration:', i, 'of', num_ab_pairs)
    
    if file_name is None:
        file_name = 'vdp_data_' + str(num_ab_pairs) + 'pts_[soln,FT,hist,phase(' + \
                    str(include_phase) + '),param]'
    
    file_path = './datafiles/' + file_name
    
    print('Writing datafile to', file_path + '.npy')
    np.save(file_path, data)
    print('Done')
    
if __name__ == '__main__':
    generate_data()    