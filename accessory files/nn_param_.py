#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 12:32:10 2019

@author: adamreidsmith
"""

'''
Neural network to predict parameters a and b given a solution to the 
Van der Pol equation: x'' = a*(1 - x^2)*x' - b*y + f(t) with t mod T1, where T1 
is the period of the function f(t).

The data file contains 3000 numerical solutions of the Van der Pol equation
with random values of a and b in [0,1).  Each solution has 2000 points in 
t = [0, 20].
'''

import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

#import pdb  #Debugging: pdb.set_trace()

def net(name='../Data_files/vdp_param_data_[[a,b],[t,x]]_tmax10.npy'):
    
    med_per_errs = []

    class Data(Dataset):
        
        def __init__(self):
            self.data = np.load(name)
            
            #Parameters used in solution of Van der Pol oscillator
            self.parameters = torch.Tensor([self.data[i][0] for i in range(len(self.data))])
            
            self.len = self.parameters.shape[0]
            
            #Soution to Van der Pol oscillator with given parameters
            #List of pairs t mod T1, x
            
            #self.vdp_soln = torch.Tensor([np.delete(self.data[i], 0, axis=0) for i in range(len(self.data))]).flatten(1)
            
            self.vdp_soln = torch.Tensor([self.data[i][1:][:,1] for i in range(len(self.data))])
                    
        def __getitem__(self, index):
            return self.vdp_soln[index], self.parameters[index]
        
        def __len__(self):
            return self.len
            
    
    dataset = Data()
    
    # Lengths of the training and validation datasets
    train_len = int(0.8*dataset.len)
    valid_len = dataset.len - train_len
    
    #Randomly split the data into training and validation datasets
    train_data, valid_data = random_split(dataset, (train_len, valid_len))
    
    batch_size = 10
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True)
    
    
    #Model of the neural network
    class Model(nn.Module):
        
        def __init__(self):
            super(Model, self).__init__()
            
            self.l1 = nn.Linear(1000, 500)
            self.l2 = nn.Linear(500, 50)
            self.l3 = nn.Linear(50, 2)
            
        def forward(self, x):
           
            x = torch.sigmoid(self.l1(x))
            x = torch.sigmoid(self.l2(x))
            return self.l3(x)
            
        
    model = Model()
    
    #Mean squared error loss
    loss_func = nn.MSELoss()
    
    #Stochastic gradient descent optimizer
    optimizer = torch.optim.Adamax(model.parameters(), lr=0.0001, weight_decay=1e-8)
    
    def evaluate():
        #Evaluation mode
        model.eval()
        for data in valid_loader:
    
            #Split batch into inputs and outputs
            x, y = data[0].squeeze(0), data[1].squeeze(0)
            
            #Forward propagation
            out = model(x)
            
            #Loss computation
            loss = loss_func(out, y)
            
            #Save training loss in this batch
            valid_loss.append(loss.item())
            
            #Compute the average percent error over a validation batch
            percent_error = 100*torch.div(abs(out - y), y)
            median_percent_error.append(np.median(percent_error.detach().numpy()))
        
        return valid_loss
        
    def train():
        #Training mode
        model.train()
        for data in train_loader:
            
            #Split batch into inputs and outputs
            x, y = data[0].squeeze(0), data[1].squeeze(0)
            
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
    
    #Training and validation loop
    n_epochs = 50
    for epoch in range(n_epochs): #An epoch is a run of the entire training dataset
    
        train_loss, valid_loss, median_percent_error = [], [], []
        
        
        if epoch % (n_epochs-1) == 0 and epoch != 0:
            #Plot a histogram of the error (predicted - true)
            error = []    
            model.eval()
            for data in valid_loader:
                #Split batch into inputs and outputs
                x, y = data[0].squeeze(0), data[1].squeeze(0)
                out = model(x)
                error.append((out - y).detach().numpy())
            
            error = np.array(error)
            a_error = np.array([error[i][j][0] for i in range(len(error)) for j in range(batch_size)])
            b_error = np.array([error[i][j][1] for i in range(len(error)) for j in range(batch_size)])
            
            plt.figure(1, figsize=(8,6))
            plt.hist(a_error, bins=30, color='b')
            plt.title('Error in parameter a')
            plt.xlabel('Predicted - True')
            plt.figure(2, figsize=(8,6))
            plt.hist(b_error, bins=30, color='k')
            plt.title('Error in parameter b')
            plt.xlabel('Predicted - True')
            plt.show()
            
    
        train_loss = train()
        
        valid_loss = evaluate()
        if __name__ == '__main__':
            print('Epoch', epoch+1)
            print('  Mean Epoch Training Loss:  ', np.mean(train_loss))
            print('  Mean Epoch Validation Loss:', np.mean(valid_loss))
            print('  Median percent error:      ', np.median(np.array(median_percent_error)), '%')
        
        if epoch >= n_epochs - 5:
            med_per_errs.append(np.median(np.array(median_percent_error)))
    
    return np.average(med_per_errs)
    
net()

   
'''
Requres a density of around 50 points per second in solution of vdp equation to
loss of 0.004 in 30 epochs with lr=0.0001.  With lower density convergence is 
slow or not achieved. For moded data, there is major overfitting. For unmoded 
data, convergence on training or validation data is not achieved.
Number of a,b pairs does not affect the convergence much.
'''



'''
TO DO:
    Switch to a single period T in the forcing function f(t).
    
    To encorporate the period, use t' = t mod T.
    
    Plot a histogram of x values from 0 to T.
    
    Machine learn with this new data encorporating the time.
    
    Find a better way of preresenting the output of the neural network and
    the associated errors.
    
    Consider two periods.
    
    Set t1 = t mod T1, t2 = t mod T2  (or something else, consider time 
    [seconds, hours, weeks] to figure out the best way to represent t1 and t2).
    
    Plot 2D histogram of the data in a similar fashion to the 1D case.
    
    Use image processing neural network techniques to analyse data.
    
    Consider a convolutional neural network.  This separates the image into
    sections as pixels on opposite sides of the NN are not related.  However,
    boundary conditions are periodic in this case, so edges are joined.
    
    
    
    
    
    Try 2D histogram with single period data. t mod T1 in one dimension, 
    x in the other.
'''


    
    
    


