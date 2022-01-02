#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 08:30:26 2019

@author: adamreidsmith
"""

'''
Standard neural network with dropout.  
Trains on data on a sphere with no phase in forcing function.
'''

import torch
import numpy as np
from torch import nn
from scipy.signal import find_peaks
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
#import pdb #Debugging: pdb.set_trace()

def main():
    
    class Data(Dataset):
        
        def __init__(self):
            print('\nLoading data...')
            self.data = np.load('../Data_files/2sphere/vdp_2sphere_800pts_nophase_2periods_[soln,[a1,a2,b1,b2]].npy')
            self.data = self.data[:4*800]

            dlen = len(self.data)
            
            #Parameters used in solution of Van der Pol oscillator
            self.parameters = torch.Tensor([self.data[i] for i in range(dlen) if (i+1) % 2 == 0])
            
            self.len = self.parameters.shape[0]
            
            #Tensor of x values in the solution of the Van der Pol equation
            self.soln = [self.data[i] for i in range(dlen) if i % 2 == 0]
            self.time = torch.Tensor(list(self.soln[0][:,0]))
            self.soln = torch.Tensor([[soln[i][1] for i in range(len(self.soln[0]))] for soln in self.soln])

            #Tensor of complex values of Fourier transform
            #self.fft = [self.data[i] for i in range(dlen) if (i-1) % 3 == 0]
            #self.fft = [fft[:,1] for fft in self.fft]
            
            #Tensor of absolute value of Fourier transform values
            #self.absfft = torch.Tensor(list(np.abs(self.fft)))
            #self.absfft /= max(self.absfft.flatten()).item()
            
            #Tensor of real part of Fourier transform values
            #self.refft = torch.Tensor(list(np.real(self.fft)))
            
            #Tensor of imaginary part of Fourier transform values
            #self.imfft = torch.Tensor(list(np.imag(self.fft)))
            
            def as_spherical(xyz):
                #Takes list xyz (single coord)
                x = np.array(xyz[:,0])
                y = np.array(xyz[:,1])
                z = np.array(xyz[:,2])
                theta = np.arccos(z)
                phi = np.arctan2(y,x)
                return np.array([theta,phi]).T
            
            #[theta, phi] coordinates of points in self.soln
            #r = 1 as points lie on a sphere
            #self.soln_spherical = torch.Tensor([as_spherical(soln) for soln in self.soln])
                        
            #Random data to test against
            #self.rand = torch.Tensor([np.random.rand(2048) for _ in range(800)])
            
        def __getitem__(self, index):            
                
            item = self.soln[index].flatten()
                                    
            return item, self.parameters[index]
            
        def __len__(self):
            return self.len
                       
    
    dataset = Data()
    
    # Lengths of the training and validation datasets
    train_len = int(0.75*dataset.len)
    valid_len = dataset.len - train_len
        
    #Randomly split the data into training and validation datasets
    train_data, valid_data = random_split(dataset, (train_len, valid_len))
    
    batch_size = 5
    
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True)
   
    
    #Model of the neural network
    class Model(nn.Module):
        
        def __init__(self):
            super(Model, self).__init__()
            
            n_inputs = 3072
            
            #Fully connected linear layers
            #self.dropout1 = nn.Dropout(p=0.4)
            self.fc1 = nn.Linear(in_features=n_inputs, out_features=500)
            self.dropout2 = nn.Dropout(p=0.4)
            self.fc2 = nn.Linear(in_features=500, out_features=50)
            self.dropout3 = nn.Dropout(p=0.2)
            self.fc3 = nn.Linear(in_features=50, out_features=4)
                       
        def forward(self, x):
            
            #Linear layers wih dropout
            #x = self.dropout1(x)
            x = torch.sigmoid(self.fc1(x))
            x = self.dropout2(x)
            x = torch.sigmoid(self.fc2(x))
            x = self.dropout3(x)
            return self.fc3(x)

        
    model = Model()
    
    #Mean squared error loss
    #loss_func = nn.MSELoss()
        
    #Log-cosh loss
    LogCoshLoss = lambda x, y: torch.log(torch.cosh(2*(x - y))).sum()
    loss_func = LogCoshLoss
    
    #Optimizer
    lr = 0.001    #Adam/Adamax: ~0.001   RMSprop: ~0.0001
    wd = 1e-8
    lr_factor = .98
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_factor)
    
    def evaluate():
        #Evaluation mode
        model.eval()
        for data in valid_loader:
    
            #Split batch into inputs and outputs
            x, y = data[0], data[1].squeeze()
                        
            #Forward propagation
            out = model(x)
            
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
            x, y = data[0], data[1].squeeze()
            
            def closure():
                #Reset gradients to zero
                optimizer.zero_grad()
                
                #Forward propagation
                out = model(x)
                        
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
            x, y = data[0], data[1].squeeze()
            
            out = model(x)
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
        
    #Number of epochs to train for
    n_epochs = 40
    
    #Print statistics about the current run
    print('\nModel Information:\n', model, sep='')
    print('\nRun Start', 
          '\n  Batch size:', batch_size, 
          '\n  Epochs:',  n_epochs,
          '\n  Training data size:', len(train_loader)*batch_size,
          '\n  Validation data size:', len(valid_loader)*batch_size,
          '\n  Learning rate:', lr,
          '\n  LR decay factor:', lr_factor,
          '\n  Weight decay:', wd,
          '\n  Loss function:', repr(loss_func).partition('(')[0],
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

#main()
    
    
    
    
    
    
    
    
    
arr = [6.299,8.05,7.05,7.17,7.77,6.91,7.02,6.96,6.76,7.33,7.23,7.09,6.38,6.97,7.23]

print(np.mean(arr))
print(np.std(arr))
