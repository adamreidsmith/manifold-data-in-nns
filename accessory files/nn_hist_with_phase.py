#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 16:24:22 2019

@author: adamreidsmith
"""

'''
Convolutional neural network with pooling and linear layers with dropout.
Input is a histogram (100x100 matrix) of t mod T vs x(t), where x(t) is a solution
to the Van der Pol equation x'' = a*(1 - x^2)*x' - b*y + f(t) and T is the period 
of f.  Here, f is given a random phase that is not incorporated into training.

The network predicts values to a and b given the histogram.
'''

import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
#import pdb #Debugging: pdb.set_trace()

import time

def main():
    
    class Data(Dataset):
        
        def __init__(self):
            self.data = np.load('../Data_files/hist/vdp_rand_phase_3000_pts_[hist,phi,[a,b]].npy')
            self.data = self.data[:3*800]
            
            #Parameters used in solution of Van der Pol oscillator
            self.parameters = torch.Tensor([self.data[i] for i in range(len(self.data)) if (i+1) % 3 == 0])
            
            self.len = self.parameters.shape[0]
            
            #List of values of 2d histograms of t mod T vs x for solutions to the Van der Pol equation
            self.vdp_soln = torch.Tensor([[self.data[i]] for i in range(len(self.data)) if i % 3 == 0])
            
        def __getitem__(self, index):
            return self.vdp_soln[index], self.parameters[index]
        
        def __len__(self):
            return self.len
                       
    
    dataset = Data()
    
    # Lengths of the training and validation datasets
    train_len = int(0.7*dataset.len)
    valid_len = dataset.len - train_len
        
    #Randomly split the data into training and validation datasets
    train_data, valid_data = random_split(dataset, (train_len, valid_len))
    
    batch_size = 2
    
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True)
        

    #Model of the neural network
    class Model(nn.Module):
        
        def __init__(self):
            super(Model, self).__init__()
            
            #Computes the matrix dimension after applying convolution or max pooling
            #Parameters: data size, kernel size, stride, padding, dialtion
            new_dim = lambda ds, ks, s=1, p=0, d=1: int((ds + 2*p - d*(ks - 1) - 1 + s)//s)
            
            #Convolution and max pooling layers
            #kernel_size: dimension of square filter (kernel) matrix
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
                    
            self.conv2 = nn.Conv2d(in_channels=6, out_channels=14, kernel_size=5)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    
            #Propagate the dimension of the data matrices through the convolution
            #max pooling layers to compute the number of in features in the first
            #fully connected linear layer.
            d1 = new_dim(new_dim(dataset.vdp_soln[0].shape[1], self.conv1.kernel_size[0]), self.pool1.kernel_size, self.pool1.stride)
            d2 = new_dim(new_dim(d1, self.conv2.kernel_size[0]), self.pool2.kernel_size, self.pool2.stride)
            self.in_f = d2*d2*self.conv2.out_channels
    
            #Fully connected linear layers: [self.in_f](dropout) -> [500](dropout) -> [50](dropout) -> [2]
            self.dropout1 = nn.Dropout(p=0.4)
            self.fc1 = nn.Linear(in_features=self.in_f, out_features=500)
            self.dropout2 = nn.Dropout(p=0.4)
            self.fc2 = nn.Linear(in_features=500, out_features=50)
            self.dropout3 = nn.Dropout(p=0.2)
            self.fc3 = nn.Linear(in_features=50, out_features=2)
            
        def forward(self, x):
            #Convolution and max pooling layers
            #Input -> Convolution -> Activation function -> Pooling
            x = self.pool1(torch.relu(self.conv1(x)))
            x = self.pool2(torch.relu(self.conv2(x)))
            
            #Reshape data from matrix to vector to transition from convolution to linear layers
            x = x.view(-1, self.in_f)
            
            #Linear layers wih dropout
            x = self.dropout1(x)
            x = torch.sigmoid(self.fc1(x))
            x = self.dropout2(x)
            x = torch.sigmoid(self.fc2(x))
            x = self.dropout3(x)
            return self.fc3(x)
        
        
    model = Model()
    
    #Mean squared error loss
    #loss_func = nn.MSELoss()
    
    #Log-cosh loss
    def LogCoshLoss(x, y):
        return torch.log(torch.cosh(2*(x - y))).sum()
    
    loss_func = LogCoshLoss
    
    #Stochastic gradient descent optimizer
    lr = 0.001    #Adam/Adamax: ~0.001   RMSprop: ~0.0001
    wd = 1e-8
    lr_factor = .95
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_factor)
    
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
            all_percent_error.extend(percent_error.flatten().squeeze(0).tolist())
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
    
    def plot_hist():
        #Plot histograms of the error (Predicted - True) in the predicted data
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
        
        plt.figure(figsize=(8,6))
        plt.hist(a_error, bins=30, color='b')
        plt.title('Prediction error in parameter \'a\' in validation data')
        plt.xlabel('Predicted - True')
        plt.figure(figsize=(8,6))
        plt.hist(b_error, bins=30, color='k')
        plt.title('Prediction error in parameter \'b\' in validation data')
        plt.xlabel('Predicted - True')
        
        plt.figure(figsize=(8,6))
        p_err_less_100 = [i for i in all_percent_error if i <= 100]
        n_more_100 = len(all_percent_error) - len(p_err_less_100)  
        plt.hist(p_err_less_100, bins=30)
        plt.text(x=plt.xlim()[1]-35, y=plt.ylim()[1]-10, s='More than 100% error:\n'+str(n_more_100))
        plt.xlabel('Percent Error')
        plt.title('Histogram of percent errors in predictions of validation data')
        
        plt.show()
    
    #Number of epochs to train for
    n_epochs = 50
    
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
    
        train_loss, valid_loss, median_percent_error, all_percent_error = [], [], [], []
        
        #Train the network
        train_loss = train()
        
        #Evaluate the network
        valid_loss = evaluate()
        
        print('Epoch:', epoch+1,
              '\n  Learning rate:             ', scheduler.get_lr()[0],
              '\n  Mean epoch training loss:  ', np.mean(train_loss),
              '\n  Mean epoch validation loss:', np.mean(valid_loss),
              '\n  Overfitting factor:        ', np.mean(valid_loss)/np.mean(train_loss),
              '\n  Median percent error:      ', np.median(np.array(median_percent_error)), '%')
        
        #Update the learing rate
        scheduler.step()
    
        
    plot_hist()

start = time.time()
main()
end = time.time()
print('time', end-start)












