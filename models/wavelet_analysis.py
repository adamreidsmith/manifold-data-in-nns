#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 10:27:35 2019

@author: adamreidsmith
"""

'''
Module for applying wavelet analysis to the solutions of the Van der Pol equation.
Used in nn_wavelet.py.
'''

import pywt
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
        
def plot_wavelet_transform(data, 
                           time, 
                           wavelet_name='morl', 
                           title='Wavelet transform',
                           plot_signal=True,
                           plot_fourier=True,
                           maxima=True,
                           maxima_min_dist=1,
                           num_maxima_peaks=np.inf,
                           n_scale_pts=150,
                           scale_min=1.66,
                           scale_max=130,
                           n_phis=3,
                           n_data_points=800):

    #Define wavelet
    wavelet = pywt.ContinuousWavelet(wavelet_name)
    
    #Plot the signal
    if plot_signal:
        fig, ax = plt.subplots(figsize=(10,2))
        ax.plot(time, data)
        ax.set_title('Signal', fontsize=16)
    
    #Plot the Fourier transform of the signal
    if plot_fourier:
        fig, ax = plt.subplots(figsize=(10,3))
        ax.plot(time, np.abs(np.fft.fft(data)))
        ax.set_title('Modulus of Fourier transform', fontsize=16)
        ax.set_yscale('log')

    #Scale points to use
    scales = np.linspace(scale_min, scale_max, n_scale_pts)
    sampling_period = time[1]-time[0]
    
    #Compute the Continuous Wavelet Transform
    coefs, freqs = pywt.cwt(data, scales, wavelet, sampling_period)
    
    #Plot the wavelet transform with a contour plot
    fig, ax = plt.subplots(figsize=(10,7))
    im = ax.contourf(time, np.log2(freqs), coefs, levels=200, cmap=plt.cm.seismic)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    ax.set_title(title, fontsize=16)
            
    #Set axes titles and labels
    xlabel = 'Time'
    ylabel = 'Frequency'
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_yticklabels(2**ax.get_yticks())
    
    #Local maxima of wavelet transform  
    if maxima:    
        local_max = peak_local_max(abs(coefs), min_distance=maxima_min_dist, num_peaks=num_maxima_peaks)
        time_of_max = [time[i] for i in local_max[:,1]]
        freq_of_max = np.log2([freqs[i] for i in local_max[:,0]])
        ax.scatter(time_of_max, freq_of_max, c='g', s=10, label='Local extrema')
        ax.legend()
    
    plt.show()

def wavelet_transform(data, 
                      time, 
                      wavelet_name='morl', 
                      n_scale_pts=150,
                      scale_min=1.66,
                      scale_max=130):
    
    #Convert to numpy arrays
    if type(data) != np.ndarray:
        data = np.array(data)
    if type(time) != np.ndarray:
        time = np.array(time)
    
    #Define wavelet
    wavelet = pywt.ContinuousWavelet(wavelet_name)
    
    #Scale points to use in transform
    scales = np.linspace(scale_min, scale_max, n_scale_pts)
    sampling_period = time[1]-time[0]
    
    #Compute and return the Continuous Wavelet Transform
    return pywt.cwt(data, scales, wavelet, sampling_period)


def wavelet_transform_maxima(wavelet_transform,
                             time,
                             min_distance=1, 
                             threshold_abs=None, 
                             threshold_rel=None, 
                             num_peaks=np.inf):
    
    #Convert to numpy arrays
    if type(wavelet_transform[0]) != np.ndarray:
        wavelet_transform[0] = np.array(wavelet_transform[0])
    if type(wavelet_transform[1]) != np.ndarray:
        wavelet_transform[1] = np.array(wavelet_transform[1])

    
    #Compute the local maxima
    local_max_indices = peak_local_max(np.abs(wavelet_transform[0]), 
                                       min_distance=min_distance, 
                                       threshold_abs=threshold_abs, 
                                       threshold_rel=threshold_rel, 
                                       num_peaks=num_peaks)
    
    #Compute the time and frequency of the local maxima
    maxima = [wavelet_transform[0][local_max_indices[i][0]][local_max_indices[i][1]] for i in range(len(local_max_indices))]
    time_of_max = [time[i] for i in local_max_indices[:,1]]
    freq_of_max = [wavelet_transform[1][i] for i in local_max_indices[:,0]]

    return local_max_indices, maxima, time_of_max, freq_of_max









