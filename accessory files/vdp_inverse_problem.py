#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 08:52:09 2019

@author: adamreidsmith
"""

import math as m
import numpy as np
import random as r
from matplotlib import pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.linalg import lstsq

def main():
    
    class Data():
        
        def __init__(self):
            print('\nLoading data...\n')
            self.data = np.load('../Data_files/vdp_2000_pts_[soln,FT,hist,phi,[a,b]].npy')
            #self.data = self.data[:5*800]
            
            dlen = len(self.data)
            
            #Parameters used in solution of Van der Pol oscillator
            self.parameters = np.array([self.data[i] for i in range(dlen) if (i+1) % 5 == 0])
                        
            #Phase phi included in the forcing function cos(wt+phi)
            self.phase = np.array([[self.data[i]] for i in range(dlen) if (i+2) % 5 == 0])
                        
            #Array of values of 2d histograms of t mod T vs x for solutions to the Van der Pol equation
            #self.hist = np.array([[self.data[i]] for i in range(dlen) if (i+3) % 5 == 0])
            
            #Tensor of absolute value of Fourier transform values
            #self.absfft = np.array([np.abs(self.data[i]) for i in range(dlen) if (i+4) % 5 == 0])
            #self.absfft = np.array([[self.absfft[i][j][1] for j in range(len(self.absfft[0]))] for i in range(len(self.absfft))])
            #self.absfft /= max(self.absfft.flatten()).item()
            
            #Tensor of real part of Fourier transform values
            #self.refft = np.array([np.real(self.data[i]) for i in range(dlen) if (i+4) % 5 == 0])
            #self.refft = np.array([[self.refft[i][j][1] for j in range(len(self.refft[0]))] for i in range(len(self.refft))])
            
            #Tensor of imaginary part of Fourier transform values
            #self.imfft = np.array([np.imag(self.data[i]) for i in range(dlen) if (i+4) % 5 == 0])
            #self.imfft = np.array([[self.imfft[i][j][1] for j in range(len(self.imfft[0]))] for i in range(len(self.imfft))])
            
            #Tensor of x values in the solution of the Van der Pol equation
            self.soln = np.array([self.data[i] for i in range(dlen) if i % 5 == 0])
            self.soln = np.array([[self.soln[i][j][1] for j in range(len(self.soln[0]))] for i in range(len(self.soln))])
            
            #Time points at which the Van dar Pol equation was evaluated.
            self.time = np.array([self.data[i] for i in range(dlen) if i % 5 == 0])
            self.time = np.array([self.time[0][i][0] for i in range(len(self.time[0]))])

            self.len = self.parameters.shape[0]
    
    
    data = Data()   
    
    
    def spline(x, y):
        interp = InterpolatedUnivariateSpline(x, y, k=5)
        first_deriv = interp.derivative(1)
        second_deriv = interp.derivative(2)
        return interp, first_deriv, second_deriv
    
    C1 = 1
    T1 = 5
    #Forcing term in Van der Pol equation
    def f(t, phi): 
        return C1*np.cos(2*np.pi/T1*t + phi)  
    
    print('Predicting parameters...\n')
    
    #Points at which to evaluate the splines
    t_max = m.ceil(max(data.time))
    n_points = 100
    #time_samples = r.sample(range(t_max), k=n_points)
    time_samples = [t_max*r.random() for _ in range(n_points)]
    
    a_pred = np.array([])
    b_pred = np.array([])
    
    #Predict parameters using interpolated univariate splines and numerical differentiation
    for i in range(data.len):
        
        #Degree 5 interpolated univariate spline and two derivatives
        s, s_prime, s_doubleprime = spline(data.time, data.soln[i])
        
        #Phase in the forcing term
        phi = data.phase[i][0]
        
        #Van dar pol equation is of the form Ax=B where x = [a,b]
        A = np.array([[(s(t)**2 - 1)*s_prime(t), s(t)] for t in time_samples])
        B = np.array([[f(t, phi) - s_doubleprime(t)] for t in time_samples])
        
        #Least squares solution to the overdetermined system for predicted values of the parameters
        predicted_parameters = lstsq(A,B)[0]

        #Predicted parameters
        a_pred = np.append(a_pred, predicted_parameters[0])
        b_pred = np.append(b_pred, predicted_parameters[1])
         
    #True parameters
    a_true = data.parameters[:,0]
    b_true = data.parameters[:,1]

    #Plot histograms of the error (Predicted - True) in the predicted data
    a_error = a_pred - a_true
    b_error = b_pred - b_true
    plt.figure(figsize=(8,6))
    plt.hist(a_error, bins=30, color='b')
    plt.title('Prediction error in parameter \'a\'')
    plt.xlabel('Predicted - True')
    plt.figure(figsize=(8,6))
    plt.hist(b_error, bins=30, color='k')
    plt.title('Prediction error in parameter \'b\'')
    plt.xlabel('Predicted - True')
    
    #Plot histogram of the percent errors in the data
    percent_err_a = 100*abs((a_pred - a_true)/a_true) 
    percent_err_b = 100*abs((b_pred - b_true)/b_true)     
    plt.figure(figsize=(8,6))
    p_err_less_100 = [i for i in percent_err_a if i <= 100] + [i for i in percent_err_b if i <= 100]
    n_more_100 = 2*len(percent_err_a) - len(p_err_less_100)  
    plt.hist(p_err_less_100, bins=30)
    plt.text(x=plt.xlim()[1]-10, y=50, s='More than 100% error:\n'+str(n_more_100))
    plt.xlabel('Percent Error')
    plt.title('Histogram of percent errors in predictions')
    
    plt.show()
    
    print('Median percent error:', np.median(percent_err_a + percent_err_b))
    #print('Mean percent error:  ', np.mean(percent_err_a + percent_err_b))


main()

















