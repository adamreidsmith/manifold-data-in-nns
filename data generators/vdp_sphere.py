#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:28:38 2019

@author: adamreidsmith
"""

'''
Create datafiles of the solution to a coupled Van der Pol equation and map them
to a sphere.

These datafiles are created for use in 'nn_2sphere.py' and 'nn_wfm_s2.py'. 
'''
  
import numpy as np
from scipy.integrate import odeint
from os import path, mkdir

###############################################################################
'''
Inputs:
    tmax:           The upper bound of the interval [0,tmax] on which to solve 
                        the Van der Pol equation.
    
    initial_cond:   Initial condition. Can be set to 'random' or a list of length 4.
    
    n_points:       The number of time steps to include in each solution.
    
    num_ab_pairs:   The number of times to solve the equation, i.e. the number 
                        of data points.
        
    n_phases:       Number of phases in the forcing funcion F. Can be either 0, 
                        1, or 2. F is affected in the following way:
                            0:  F(t) = (C1*cos(w1*t), C2*cos(w2*t))
                            1:  F(t) = (C1*cos(w1*t + phi1), C2*cos(w2*t))
                            2:  F(t) = (C1*cos(w1*t + phi1), C2*cos(w2*t + phi2))
    
    C:              Coefficients of the terms in the forcing function. Must be 
                        a list of length 2.
    
    T:              Periods of the terms in the forcing function. Must be a list 
                        of length 2.
    
    file_name:      Name of the datafile.
'''
###############################################################################

def generate_data(tmax=500,
                  initial_cond='random',
                  n_points=2**10,
                  num_ab_pairs=800,
                  n_phases=0,
                  C=[1,np.sqrt(2)],
                  T=[5,10*np.sqrt(2)-2],
                  file_name=None):
    
    twopi = 2*np.pi
    
    #Create a directory to store datafiles if it doesn't aready exist
    if not path.exists('./datafiles'):
        mkdir('./datafiles')
    
    assert type(C) == type(T) == list and 2 == len(C) == len(T), \
           'C and T must be lists of length 2.'
    
    #RHS
    def f(t, phi, C, T):
        return C[0]*np.cos(twopi/T[0]*t) if n_phases == 0 else C[0]*np.cos(twopi/T[0]*t + phi[0])
    
    def g(t, phi, C, T):
        return C[1]*np.cos(twopi/T[1]*t + phi[1]) if n_phases == 2 else C[1]*np.cos(twopi/T[1]*t)
    
    data = []
    
    for i in range(num_ab_pairs):
        a1, a2 = np.random.rand(), np.random.rand() #Random numbers in [0,1)
        b1, b2 = np.random.rand(), np.random.rand() #Random numbers in [0,1)
        
        phi = []
        if n_phases == 1:
            phi.append(twopi*np.random.rand())
        elif n_phases == 2:
            for _ in range(2):
                phi.append(twopi*np.random.rand())
        
        if initial_cond == 'random':
            ic = [2*np.random.rand() - 1, 
                  2*np.random.rand() - 1, 
                  2*np.random.rand() - 1, 
                  2*np.random.rand() - 1]
        else:
            ic = initial_cond
    
        #Van der Pol oscillator equation
        def vanderpol(ic,t):
            x1 = ic[0]
            x2 = ic[1]
            y1 = ic[2]
            y2 = ic[3]
            y1d = f(t, phi, C, T) + a1*(1 - x1**2 - x2**2)*y1 - b1*x1
            y2d = g(t, phi, C, T) + a2*(1 - x1**2 - x2**2)*y2 - b2*x2
            x1d = y1
            x2d = y2
            return [x1d, x2d, y1d, y2d]
        
        #Solve the ivp numerically    
        npoints = 10*2**10
        tfull = np.linspace(0,tmax,npoints)
        sol = odeint(vanderpol, ic, tfull)
        x1full, x2full= sol[:,0], sol[:,1]
        
        #Keep every tenth data point
        indices = [i for i in range(npoints) if i % 10 == 0]
        t = np.array([tfull[i] for i in indices])
        x1 = [x1full[i] for i in indices]
        x2 = [x2full[i] for i in indices]  
        
        #Map to sphere
        def to_sphere(x1,x2):
            r = []
            for i in range(len(x1)):
                x = np.sin(x1[i])*np.cos(x2[i])
                y = np.sin(x1[i])*np.sin(x2[i])
                z = np.cos(x1[i])
                r.append([x,y,z])
            return np.array(r)
            
        r = to_sphere(x1,x2)
    
        soln = np.array([[t[i],r[i]] for i in range(len(t))])
            
        data.append(soln)
        data.append(phi)
        data.append([a1,a2,b1,b2]) 
        
        if i % 10 == 0 and __name__ == '__main__':
            print('Iteration:', i, 'of', num_ab_pairs)
            
    if file_name is None:
        file_name = 'vdp_2sphere_' + str(num_ab_pairs) + 'pts_[soln,phase(' + str(n_phases) + '),param]'
    
    file_path = './datafiles/' + file_name
    
    print('Writing datafile to', file_path + '.npy')
    np.save(file_path, data)
    print('Done')

if __name__ == '__main__':
    generate_data()








