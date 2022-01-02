#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 12:28:24 2019

@author: adamreidsmith
"""

'''
Standard neural network with dropout. Trains on data on a 2-sphere.
'''

#Path of the datafile created by 'vdp_sphere.py'.
file_path = './datafiles/vdp_2sphere_800pts_[soln,phase(2),param].npy'

from os import path
assert path.exists(file_path), 'Datafile not found. Please run \'vdp_sphere.py\' \
                               to generate a datafile.'
import torch
import numpy as np
from torch import nn
from scipy.signal import find_peaks
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

###############################################################################
'''
Inputs:
    n_epochs:           Number of epochs to train for.
    
    batch_size:         Batch size.
    
    lr:                 Learning Rate.
    
    weight_decay:       Weight decay factor.
    
    lr_factor:          Learning rate decay factor.  Learning rate is multiplied 
                            by this factor every epoch.
                            
    loss_function:      Loss function. Can be:
                            'mean square':   loss = sum((x_i - y_i)^2)
                            'log cosh':      loss = sum(log(cosh(x_i - y_i)))
    
    num_peaks:          Number of peaks in the Fourier transform to train on.
    
    net_input:          Dataset to train on. Can be:
                            'soln':     Network is trained directly on the solution 
                                        points.
                            'peaks':    Network is trained on the peaks of the 
                                        Fourier transform of the stereographic
                                        projection of the data.
'''
###############################################################################

def main(n_epochs=40,
         batch_size=5,
         lr=0.001,
         weight_decay=1e-8,
         lr_factor=0.98,
         loss_function='log cosh',
         num_peaks=3,
         net_input='peaks'):
    
    class Data(Dataset):
        
        def __init__(self):
            print('\nLoading data...')
            self.data = np.load(file_path)
            
            dlen = len(self.data)
            
            #Parameters used in solution of Van der Pol oscillator
            self.parameters = torch.Tensor([self.data[i] for i in range(dlen) if (i+1) % 3 == 0])
            
            self.len = self.parameters.shape[0]
            
            #Phase(s) phi included in the forcing function
            self.phase = torch.Tensor([self.data[i] for i in range(dlen) if (i+2) % 3 == 0])
            
            #Tensor of x values in the solution of the Van der Pol equation
            self.soln = [self.data[i] for i in range(dlen) if i % 3 == 0]
            self.time = torch.Tensor(list(self.soln[0][:,0]))
            
            self.soln = torch.Tensor([[soln[i][1] for i in range(len(self.soln[0]))] for soln in self.soln])
            
            #Indices of the top n peaks in each Fourier transform
            def get_max_peaks(num_max_peaks, absfft, peaks_indices):
                
                max_n_peaks_indices = []
                for i in range(len(peaks_indices)):
                    for j in range(num_max_peaks):
                        key_func = lambda index: absfft[i][index]
                        try:
                            max_peak_index = max(peaks_indices[i], key=key_func)
                        except:
                            max_peak_index = None
                        if j == 0:
                            appendee = [max_peak_index] if max_peak_index is not None else [max_n_peaks_indices[-1][0]]
                            max_n_peaks_indices.append(appendee)
                        else:
                            appendee = max_peak_index if max_peak_index is not None else max_n_peaks_indices[-1][0]
                            max_n_peaks_indices[-1].append(appendee)                        
                        
                        index = np.argwhere(peaks_indices[i] == max_peak_index)
                        peaks_indices[i] = np.delete(peaks_indices[i], index)
                
                #Values and frequencies of top n peaks in each Fourier transform
                max_n_peaks = [[absfft[i][j] for j in max_n_peaks_indices[i]] for i in range(len(absfft))]
                max_n_peaks_time = [[self.time[j].item() for j in max_n_peaks_indices[i]] for i in range(len(absfft))]
                    
                return torch.Tensor(max_n_peaks_indices), torch.Tensor(max_n_peaks), torch.Tensor(max_n_peaks_time)
                        
            def stereographic_projection(XYZ):
                XYZ = np.array(XYZ)
                oneminusz = XYZ[:,2] + 1
                xcoord = XYZ[:,0]/oneminusz
                ycoord = XYZ[:,1]/oneminusz
                coords = np.empty((xcoord.size, 2))
                coords[:,0] = xcoord
                coords[:,1] = ycoord
                return coords
            
            print('Computing stereographic projections...')
            
            #Stereographic projection and Fourier transform of each component
            self.stereo_proj = [stereographic_projection(soln) for soln in self.soln]
            self.stereo_ftx = np.array([np.fft.fft(stereo_proj[:,0])[:len(self.time)//2+1] for stereo_proj in self.stereo_proj])
            self.stereo_fty = np.array([np.fft.fft(stereo_proj[:,1])[:len(self.time)//2+1] for stereo_proj in self.stereo_proj])
            
            print('Computing peaks of Fourier transforms...')
            
            #Peaks of the Fourier transforms of the stereographic projection
            stereo_peaks_indices_x = [find_peaks(np.abs(ft))[0] for ft in self.stereo_ftx]
            stereo_peaks_indices_y = [find_peaks(np.abs(ft))[0] for ft in self.stereo_fty]
            self.num_max_peaks = num_peaks
            _, self.stereo_ftx_peak_values, self.stereo_ftx_peak_times = get_max_peaks(self.num_max_peaks, np.abs(self.stereo_ftx), stereo_peaks_indices_x)
            _, self.stereo_fty_peak_values, self.stereo_fty_peak_times = get_max_peaks(self.num_max_peaks, np.abs(self.stereo_fty), stereo_peaks_indices_y)
            
             
        def __getitem__(self, index):
            if net_input == 'peaks':
                item = torch.cat((self.stereo_ftx_peak_values[index], self.stereo_ftx_peak_times[index],
                                  self.stereo_fty_peak_values[index], self.stereo_fty_peak_times[index]))
            elif net_input == 'soln':
                item = self.soln[index].flatten(0)
            else:
                raise RuntimeError('net_input is not recognized. Set net_input \
                                   to \'soln\' or \'peaks\'.')
                                    
            return [item, self.phase[index]], self.parameters[index]
            
        __len__ = lambda self: self.len
                       
    
    dataset = Data()
    
    # Lengths of the training and validation datasets
    train_len = int(0.75*dataset.len)
    valid_len = dataset.len - train_len
        
    #Randomly split the data into training and validation datasets
    train_data, valid_data = random_split(dataset, (train_len, valid_len))
        
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True)
 
    
    #Model of the neural network
    class Model(nn.Module):
        
        def __init__(self):
            super(Model, self).__init__()
            
            if net_input == 'peaks':
                n_inputs = 4*dataset.num_max_peaks + len(dataset.phase[0])
            else:
                n_inputs = 3*len(dataset.soln[0])+ len(dataset.phase[0])

            #Fully connected linear layers
            #self.dropout1 = nn.Dropout(p=0.4)
            self.fc1 = nn.Linear(in_features=n_inputs, out_features=500)
            self.dropout2 = nn.Dropout(p=0.4)
            self.fc2 = nn.Linear(in_features=500, out_features=50)
            self.dropout3 = nn.Dropout(p=0.2)
            self.fc3 = nn.Linear(in_features=50, out_features=4)
                       
        def forward(self, x, phi):
                        
            #Append phi to the input vector
            x = torch.cat((x,phi),1)
            
            #Linear layers wih dropout
            #x = self.dropout1(x)
            x = torch.sigmoid(self.fc1(x))
            x = self.dropout2(x)
            x = torch.sigmoid(self.fc2(x))
            x = self.dropout3(x)
            return self.fc3(x)

        
    model = Model()
    
    if loss_function == 'mean square':
        loss_func = nn.MSELoss()
    elif loss_function == 'log cosh':
        loss_func = lambda x, y: torch.log(torch.cosh(2*(x - y))).sum()
    else:
        raise RuntimeError('loss_function not recognized. \
                           Set loss_function to \'mean square\' or \'log cosh\'')    
    
    #Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_factor)
    
    def evaluate():
        #Evaluation mode
        model.eval()
        for data in valid_loader:
    
            #Split batch into inputs and outputs
            x, phi, y = data[0][0], data[0][1], data[1].squeeze()
                        
            #Forward propagation
            out = model(x, phi)
            
            #Loss computation
            loss = loss_func(out, y)
            
            #Save training loss in this batch
            valid_loss.append(loss.item())
            
            #Compute the average percent error over a validation batch
            percent_error = 100*torch.div(abs(out - y), y)
            all_percent_error.extend(percent_error.flatten().squeeze(0).tolist())
        
        return valid_loss
        
    def train():
        #Training mode
        model.train()
        for data in train_loader:
            
            #Split batch into inputs and outputs
            x, phi, y = data[0][0], data[0][1], data[1].squeeze()
            
            def closure():
                #Reset gradients to zero
                optimizer.zero_grad()
                
                #Forward propagation
                out = model(x, phi)
                        
                #Loss computation
                loss = loss_func(out, y)
                
                #Backpropagation
                loss.backward()
                
                return loss
            
            #Weight optimiation
            optimizer.step(closure)
            
            #Save training loss in this batch
            train_loss.append(closure().item())
            
        return train_loss
    
    def plot_hist():
        #Plot histograms of the error (Predicted - True) in the predicted data
        error = []    
        model.eval()
        for data in valid_loader:
            #Split batch into inputs and outputs
            x, phi, y = data[0][0], data[0][1], data[1].squeeze()
            
            out = model(x, phi)
            error.append((out - y).detach().numpy())
                
        error = np.array(error)

        params = ['a1','a2','b1','b2']
        colors = ['b','k','r','g']
        for i in range(4):
            relevant_error = np.array([error[j][k][i] for j in range(len(error)) for k in range(batch_size)])
        
            plt.figure(figsize=(8,6))
            plt.hist(relevant_error, bins=30, color=colors[i])
            plt.title('Prediction error in parameter \'' + params[i] + '\' in validation data')
            plt.xlabel('Predicted - True')
        
        plt.figure(figsize=(8,6))
        p_err_less_100 = [i for i in all_percent_error if i <= 100]
        n_more_100 = len(all_percent_error) - len(p_err_less_100)  
        plt.hist(p_err_less_100, bins=30)
        plt.text(x=plt.xlim()[1]-35, y=plt.ylim()[1]-20, s='More than 100% error:\n'+str(n_more_100))
        plt.xlabel('Percent Error')
        plt.title('Histogram of percent errors in predictions of validation data')
        
        plt.show()
    
    #Print statistics about the current run
    print('\nModel Information:\n', model, sep='')
    print('\nRun Start', 
          '\n  Batch size:', batch_size, 
          '\n  Epochs:',  n_epochs,
          '\n  Training data size:', len(train_loader)*batch_size,
          '\n  Validation data size:', len(valid_loader)*batch_size,
          '\n  Learning rate:', lr,
          '\n  LR decay factor:', lr_factor,
          '\n  Weight decay:', weight_decay,
          '\n  Loss function:', loss_function,
          '\n  Optimizer:', repr(optimizer).partition('(')[0],
          '\n  LR scheduler:', repr(scheduler)[repr(scheduler).find('er.')+3:repr(scheduler).find(' obj')],
          '\n')
    
    #Training and evaluation loop
    for epoch in range(n_epochs): #An epoch is a run of the entire training dataset
    
        train_loss, valid_loss, all_percent_error = [], [], []
        
        #Train the network
        train_loss = train()

        #Evaluate the network
        valid_loss = evaluate()
        
        if (epoch+1) % 1 == 0:
            print('Epoch:', epoch+1,
                  '\n  Learning rate:             ', scheduler.get_lr()[0],
                  '\n  Mean epoch training loss:  ', np.mean(train_loss),
                  '\n  Mean epoch validation loss:', np.mean(valid_loss),
                  '\n  Overfitting factor:        ', np.mean(valid_loss)/np.mean(train_loss),
                  '\n  Median percent error:      ', np.median(np.array(all_percent_error)), '%')
        
        #Update the learing rate
        scheduler.step()
    
    if n_epochs:         
        plot_hist()
    
main()





