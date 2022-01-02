#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 14:35:43 2019

@author: adamreidsmith
"""

'''
ManifoldNet neural network based of an algorithm presented here:
    https://arxiv.org/abs/1809.06211
Trains on data on a 2-sphere.
'''

#Path of the datafile created by 'vdp_sphere.py'.
file_path = './datafiles/vdp_2sphere_800pts_[soln,phase(2),param].npy'

from os import path
assert path.exists(file_path), 'Datafile not found. Please run \'vdp_sphere.py\' \
                               to generate a datafile.'
import torch
import numpy as np
from torch import nn
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from torch import autograd

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
'''
###############################################################################

def main(n_epochs=15,
         batch_size=4,
         lr=0.001,
         lr_factor=0.98,
         weight_decay=1e-8,
         loss_function='abs'):
    
    class Data(Dataset):
        
        def __init__(self):
            print('\nLoading data...')
            self.data = np.load(file_path)

            dlen = len(self.data)
            
            #Parameters used in solution of Van der Pol oscillator
            self.parameters = torch.Tensor([self.data[i] for i in range(dlen) if (i+1) % 3 == 0])
            
            self.len = self.parameters.shape[0]
            
            #Phase phi included in the forcing function cos(wt+phi)
            self.phase = torch.Tensor([self.data[i] for i in range(dlen) if (i+2) % 3 == 0])
            
            #Tensor of x values in the solution of the Van der Pol equation
            self.soln = [self.data[i] for i in range(dlen) if i % 3 == 0]
            self.time = torch.Tensor(list(self.soln[0][:,0]))
            self.soln = torch.Tensor([[soln[i][1] for i in range(len(self.soln[0]))] for soln in self.soln])            
             
        def __getitem__(self, index):  
            
            item = self.soln[index]
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
    
    #Weighted Frechet mean
    def wFM(X, w):
        #inputs: X: tensor of points in R^3 on a 2-sphere
        #        w: tensor of weights corresponding to each point in X
        if len(w) > len(X):
            w = w[:len(X)]
        M = X[0]
        for i in range(1,len(X)):
            M = twosphere_geodesic(M, X[i], torch.div(w[i], w[:i+1].sum()))
        return M
       
    #Geodesic curve on a 2-sphere
    def twosphere_geodesic(x1,x2,t):
        
        dot = torch.dot(x1,x2)
                
        y1 = x2 - dot*x1
        y2 = y1/torch.norm(y1)
        
        c = t*torch.acos(dot)
        
        a = torch.cos(c)
        b = torch.sin(c)
        
        return torch.mul(a, x1) + torch.mul(b, y2)
        
    def wFM_batch(X, w):
        M = X[:,0]
        for i in range(1, len(X[0])):
            M = twosphere_geodesic_batch(M, X[:,i], torch.div(w[i], w[:i+1].sum()))
        assert torch.isnan(M).sum().item() == 0, '\'nan\' value encountered for M in wFM'
        return M
    
    def twosphere_geodesic_batch(x1,x2,t):
        
        d = torch.sum(x1*x2, dim=1)  # dot product of x1, x2
        
        sz = d.shape[0]
        
        y1 = x2 - d.view(sz,1)*x1
        normy1 = torch.sqrt(torch.sum(y1**2, dim=1))
        y2 = torch.div(y1, normy1.view(sz,1))
        
        c = t*torch.acos(d)
        
        a = torch.cos(c).view(sz,1)
        b = torch.sin(c).view(sz,1)
        
        return torch.mul(a, x1) + torch.mul(b, y2)
    
    def gcdist(x1,x2):
        #Great circle distance between points on a 2-sphere
        #Either x1 and x2 are the same shape, or x1 is a single point and x2 is a tensor of points
        dot = torch.sum(x1*x2, dim=1)
        return torch.acos(dot)
        
    
    #Custom convolutional layer based on the weighted Frechet mean
    class wFMLayer(nn.Module):
        
        def __init__(self, out_channels, kernel_size=3, stride=1):
            super(wFMLayer, self).__init__()
            self.kernel_size = kernel_size
            self.stride = stride
            self.out_channels = out_channels
            
            self.weights = Parameter(torch.Tensor(out_channels, kernel_size))
            self.reset_parameters()
            
        def reset_parameters(self):
            torch.nn.init.kaiming_uniform_(self.weights, a=np.sqrt(5))
            self.weights.data = self.weights.data / torch.max(self.weights.data)
            self.weights.data = self.weights.data * torch.sign(self.weights.data)
            
        def forward(self, x):
            for k in range(len(x)):
                X = [x[k][m:m + self.kernel_size].tolist() for m in range(0, len(x[k]), self.stride) if len(x[k][m:m + self.kernel_size]) == self.kernel_size]
                X.append(x[k][-self.kernel_size:].tolist())
                X = torch.Tensor(X)
                
                if k == 0:
                    batch_wFM_convolutions = torch.empty(len(x), self.out_channels, len(X), 3)
                    
                wFM_convolutions = torch.empty(self.out_channels, len(X), 3)
                
                for i in range(self.out_channels):                                        
                    wFM_convolutions[i] = wFM_batch(torch.Tensor(X), self.weights[i])
            
                batch_wFM_convolutions[k] = wFM_convolutions
                        
            return batch_wFM_convolutions
    
    class LastLayer(nn.Module):
        
        def __init__(self, xshape, out_channels):
            super(LastLayer, self).__init__()
            self.xshape = xshape
            self.FCLayer = nn.Linear(xshape[1]*xshape[2], out_channels)
        
        def forward(self, x):
            O = torch.Tensor(self.xshape[0],self.xshape[1]*self.xshape[2])
            for k in range(len(x)):
                #for each item in batch
                #x[k] consists of all Z_i's in a batch item. Each Z_i is a list of points on sphere
                
                X = x[k].reshape(self.xshape[1]*self.xshape[2], self.xshape[3])
                lenX = X.shape[0]
                M_u = wFM(X, (1/lenX)*torch.ones(lenX))
                O_i = gcdist(M_u, X)
                O[k] = O_i
                
            return self.FCLayer(O)
                
           
    #Model of the neural network
    class Model(nn.Module):
        
        def __init__(self):
            super(Model, self).__init__()
            
            wfmoc = 2
            self.wfmconv = wFMLayer(out_channels=wfmoc, kernel_size=7, stride=2)
            #self.wfmpooling = wFMPooling(kernel_size=3, stride=3)
            self.LL = LastLayer(xshape=[batch_size,wfmoc,510,3], out_channels=4)
                                   
        def forward(self, x, phi):
            
            x = self.wfmconv(x)  
            
            return self.LL(x)
        
        def normalize_weights(self, weights):
            weights = torch.div(weights, torch.max(weights))
            return weights * torch.sign(weights)

        
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
            x, phi, y = data[0][0], data[0][1], data[1].squeeze()
                        
            #Forward propagation
            out = model(x, phi)
            
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
        for data in train_loader:

            #Split batch into inputs and outputs
            x, phi, y = data[0][0], data[0][1], data[1].squeeze()
            #torch.set_anomaly_enabled(False)
            def closure():
                #Reset gradients to zero
                optimizer.zero_grad()
                
                #Forward propagation
                out = model(x, phi)
                
                #Loss computation
                loss = loss_func(out, y)
                
                #Backpropagation
                try:
                    with autograd.detect_anomaly():
                        loss.backward()
                except:
                    pass
                
                return loss
            
            #with autograd.detect_anomaly():
            optimizer.step(closure)
            
            if torch.min(model.wfmconv.weights.data) < 0 or torch.max(model.wfmconv.weights.data) > 1:
                #Keep weights in [0,1]
                model.wfmconv.weights.data = model.normalize_weights(model.wfmconv.weights.data) 

            #Save training loss in this batch
            l = loss_func(model(x, phi), y)
            if not torch.isnan(l) and not torch.isinf(l):
                train_loss.append(l.item())
        
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
