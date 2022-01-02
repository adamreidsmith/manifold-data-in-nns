#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 1 10:13:27 2019

@author: adamreidsmith
"""

'''
ManifoldNet neural network based of an algorithm presented here:
    https://arxiv.org/abs/1809.06211  
Trains on SO(3) matrices.
'''

import torch
import numpy as np
from torch import nn
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from torch import autograd
from so3_data_generation import generate_data

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
                            'abs':           loss = sum(|x_i - y_i|)
    
    n:                  Number of time series to generate. See 'so3_data_generation.py'.
    
    tn:                 Number of SO(3) matrics in each time series. See 'so3_data_generation.py'.
    
    noise_level:        See 'so3_data_generation.py'.
'''
###############################################################################

def main(n_epochs=15,
         batch_size=5,
         lr=0.001,
         lr_factor=0.95,
         weight_decay=1e-8,
         loss_function='abs',
         n=1000,
         tn=500,
         noise_level=1e-4):
    
    class Data(Dataset):
        
        def __init__(self):
            print('\nGenerating data...')
            
            self.I = [self.make_moments(0.1, 2) for _ in range(n)]
            
            self.data = [generate_data(self.I[i], 1, tn, noise_level) for i in range(n)]
            self.data = torch.Tensor(self.data)
            self.len = self.data.shape[0]
            
            self.I = torch.Tensor(self.I)
            
            self.I_reduced = [[torch.div(I[0],I[2]), torch.div(I[1],I[2])] for I in self.I]
            self.I_reduced = torch.Tensor(self.I_reduced)
        
        
        __getitem__ = lambda self, index: (self.data[index], self.I_reduced[index])
            
        __len__ = lambda self: self.len
        
        def make_moments(self, minimum, maximum):
            #Moments of inertia (can be any non-negative real number)
            I1 = np.random.uniform(minimum,maximum)
            I2 = np.random.uniform(minimum,maximum)
            I3 = np.random.uniform(minimum,maximum)
            
            while I1+I2<I3 or I1+I3<I2 or I2+I3<I1:
                I3 = np.random.uniform(minimum,maximum)
    
            return [I1,I2,I3]
    
    dataset = Data()
    
    # Lengths of the training and validation datasets
    train_len = int(0.75*dataset.len)
    valid_len = dataset.len - train_len
        
    #Randomly split the data into training and validation datasets
    train_data, valid_data = random_split(dataset, (train_len, valid_len))
        
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True)
    
    #Weighted Frechet mean
    def wFM(X, w):
        #Weighted frechet mean of SO3 matrices
        #X can be a list of matrices or a list of lists of matrices
        if len(X.shape) == 4:
            M = X[:,0]
            for i in range(1,X.shape[1]):
                M = so3_geodesic(M, X[:,i], torch.div(w[i], w[:i+1].sum()))
            return M
        else:
            M = X[0]
            for i in range(1,X.shape[0]):
                M = so3_geodesic(M, X[i], torch.div(w[i], w[:i+1].sum()))
            return M
        
    def powerm(A,k):
        #Computes matrix power A^k where k is a non-negative integer
        #A can be a square matrix or a batch of square matrices
        if k == 0 and len(A.shape) == 2:
            return torch.eye(A.shape[-1])
        
        if k == 0 and len(A.shape) == 3:
            inter = torch.eye(A.shape[-1]).unsqueeze(0)
            return inter.repeat(A.shape[0],1,1)
        
        if k == 1:
            return A
        
        P = torch.matmul(A,A)
        for i in range(k-2):
            P = torch.matmul(P,A)
        return P
                
    def expm(A, tol=1e-8, maxit=20):
        #Matrix exponential using power series definition
        #A can be a square matrix or a batch of square matrices
        shape = A.shape
        batched = False if len(shape) == 2 else True
        eps = 10.0
        exp = powerm(A,0)
        k = 1
        while eps > tol and k < maxit:
            old_exp = exp.clone()
            exp += torch.mul(1/np.math.factorial(k), powerm(A,k))
            k += 1
            
            if batched:
                inter1 = exp - old_exp
                inter2 = torch.mul(inter1, inter1).view(shape[0], shape[1]*shape[2])
                eps = torch.max(torch.sqrt(inter2).sum(dim=1))
            else:
                eps = torch.norm(exp - old_exp)
            
        return exp
    
    def trace(A):
        #Trace of a matrix or a batch of matrices
        shape = A.shape
        if len(shape) == 2:
            return torch.trace(A)
        mask = torch.eye(shape[-1]).unsqueeze(0).repeat(shape[0],1,1)
        return torch.mul(mask, A).view(shape[0], shape[1]*shape[2]).sum(dim=1)
    
    def so3_geodesic(A1,A2,t):
        #Parameterized geodesic curve in SO(3)
        #A1 and A2 can be square matrices of the same size
        #or batches of square matrices of equal size
        P = torch.matmul(torch.inverse(A1), A2)
        
        G = logm(P)
            
        O = torch.matmul(A1, expm(torch.mul(t, G)))
        
        return O
    
    def logm(A):
        #Matrix logarithm
        #A can be a square matrix or a batch of square matrices
        shape = A.shape
        
        theta = torch.acos(torch.div(trace(A) - 1, 2))
        C = torch.div(theta, torch.mul(2, torch.sin(theta)))
        
        if len(shape) == 2:
            G = torch.mul(C, A - A.transpose(0,1))
        else:
            inter = A - A.transpose(1,2)
            G = torch.mul(C.unsqueeze(1), inter.view(shape[0], shape[1]*shape[2]))
            G = G.view(shape)
        
        return G
    
    def geo_dist(A,B):
        #Geodesic distance between SO(3) matrices
        #Either x1 and x2 are the same shape, or x1 is a single point and x2 is a tensor of points
        C = torch.matmul(A.t(), B)
        return torch.abs(torch.acos((trace(C) - 1)/2))

    
    #Custom convolutional layer based on the weighted Frechet mean
    class wFMLayer(nn.Module):
        
        def __init__(self, in_channels=1, out_channels_per_ic=1, kernel_size=3, stride=1):
            super(wFMLayer, self).__init__()
            self.kernel_size = kernel_size
            self.stride = stride
            self.in_channels = in_channels
            self.out_channels_per_ic = out_channels_per_ic
            
            self.weights = Parameter(torch.Tensor(in_channels, out_channels_per_ic, kernel_size))
            self.reset_parameters()
            
        def reset_parameters(self):
            torch.nn.init.kaiming_uniform_(self.weights, a=np.sqrt(5))
            self.weights.data = self.weights.data / torch.max(self.weights.data)
            self.weights.data = self.weights.data * torch.sign(self.weights.data)
            
        def forward(self, x):
            #x has shape (batch_size, in_channels, tn, 3, 3)
            for k in range(x.shape[0]):
                for j in range(self.in_channels):
                    #Partition x into batches of size self.kernel_size
                    X = [x[k][j][m:m + self.kernel_size].tolist() for m in range(0, len(x[k][j]), self.stride) 
                         if len(x[k][j][m:m + self.kernel_size]) == self.kernel_size]

                    inter = x[k][j][-self.kernel_size:].tolist()
                    if inter != X[-1]:
                        X.append(x[k][j][-self.kernel_size:].tolist())
                    X = torch.Tensor(X)
                    
                    #X has shape [len(X), self.kernel_size, 3, 3]
                    #len(X) depends on the shape of x relative to self.kernel_size
                    #X can be thought of as a list of lists of 3x3 matrices
                    
                    if k == 0 and j == 0:
                        wFM_convolutions = torch.empty(x.shape[0], self.in_channels*self.out_channels_per_ic, len(X), 3, 3)
                        
                    for i in range(self.out_channels_per_ic):
                        wFM_convolutions[k][j * self.out_channels_per_ic + i] = wFM(X, self.weights[j][i])
 
            #wFM_convolitions has shape (batch_size, self.in_channels*self.out_channels_per_ic, len(X), 3, 3)
            return wFM_convolutions
        
    
    class LastLayer(nn.Module):
        
        def __init__(self, xshape, out_channels):
            super(LastLayer, self).__init__()
            self.xshape = xshape
            self.FCLayer = nn.Linear(xshape[1]*xshape[2], out_channels)
        
        def forward(self, x):
            O = torch.Tensor(self.xshape[0], self.xshape[1]*self.xshape[2])

            for k in range(x.shape[0]):
                #x[k] consists of all Z_i's in a batch item. Each Z_i is a list of points on sphere
                X = x[k].view(x.shape[1]*x.shape[2], x.shape[3], x.shape[4])

                lenX = X.shape[0]
                
                M_u = wFM(X, (1/lenX)*torch.ones(lenX))
                
                O_i = geo_dist(M_u, X)

                O[k] = O_i
                
            return self.FCLayer(O)
                
           
    #Model of the neural network
    class Model(nn.Module):
        
        def __init__(self):
            super(Model, self).__init__()
            
            self.wfmconv1 = wFMLayer(in_channels=1, out_channels_per_ic=2, kernel_size=5, stride=2)
            
            self.LL = LastLayer(xshape=[batch_size,2,249,3,3], out_channels=2)
                                   
        def forward(self, x):
            
            x = self.wfmconv1(x)
            
            return self.LL(x)
        
        def normalize_weights(self, weights):
            weights = torch.div(weights, torch.max(weights))
            return weights * torch.sign(weights)
        
        def check_normalize(self, weights):
            if torch.min(weights) < 0 or torch.max(weights) > 1:
                return self.normalize_weights(weights)
            return weights
        
    model = Model()
    
    if loss_function == 'mean square':
        loss_func = nn.MSELoss()
    elif loss_function == 'log cosh':
        loss_func = lambda x, y: torch.log(torch.cosh(2*(x - y))).sum()
    elif loss_function == 'abs':
        loss_func = lambda x, y: torch.abs(x-y).sum()
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
            x, y = data[0], data[1]
            #x is a batch of time series of matrices
            #y is a batch of corresponding principle moments of inertia
                        
            #Forward propagation
            out = model(x)
            
            #Loss computation
            loss = loss_func(out, y)
            
            #Save training loss in this batch
            if not torch.isnan(loss) and not torch.isinf(loss):
                valid_loss.append(loss.item())
            
            #Compute the average percent error over a validation batch
            if torch.isnan(out).sum().item() == 0 and torch.isinf(out).sum().item() == 0:
                percent_error = 100*torch.div(torch.abs(out - y), y)
                all_percent_error.extend(percent_error.flatten().squeeze(0).tolist())
        
        return valid_loss
        
    def train():
        #Training mode
        model.train()
        assert len(train_loader) % batch_size == 0, '\'train_loader\' must have length divisible by \'batch_size\''
        assert len(valid_loader) % batch_size == 0, '\'valid_loader\' must have length divisible by \'batch_size\''

        for data in train_loader:

            #Split batch into inputs and outputs
            x, y = data[0], data[1]
            #x is a batch of time series of matrices
            #x has shape (batch_size, in_channels, tn, 3, 3)
            #y is a batch of corresponding principle moments of inertia
            
            #torch.set_anomaly_enabled(False)
            def closure():
                #Reset gradients to zero
                optimizer.zero_grad()
                
                #Forward propagation
                out = model(x)
                
                #Loss computation
                loss = loss_func(out, y)
                
                #Backpropagation
                try:
                    with autograd.detect_anomaly():
                        loss.backward()
                except:
                    pass
                
                return loss
                                    
            optimizer.step(closure)
             
            #Keep weights in [0,1]
            model.wfmconv1.weights.data = model.check_normalize(model.wfmconv1.weights.data)
            
            #Save training loss in this batch
            l = loss_func(model(x), y)
            if not torch.isnan(l) and not torch.isinf(l):
                train_loss.append(l.item())
        
        return train_loss
    
    def plot_hist():
        #Plot histograms of the error (Predicted - True) in the predicted data
        error = []    
        model.eval()
        for data in valid_loader:
            #Split batch into inputs and outputs
            x, y = data[0], data[1]
            
            out = model(x)
            error.append((out - y).detach().numpy())
                
        error = np.array(error)

        params = ['I1/I3','I2/I3']
        colors = ['b','r']
        for i in range(2):
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
