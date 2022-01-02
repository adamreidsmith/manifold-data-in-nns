#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 13:49:40 2019

@author: adamreidsmith
"""

'''
Function approximation of the solution to the Van der Pol equation.
'''

import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt


class VDPData(Dataset):
    def __init__(self):
        self.data = np.loadtxt('../Data_files/vdp_data.csv')
        self.len = self.data.shape[0]
        self.t = torch.from_numpy(self.data[:,0])
        self.x = torch.from_numpy(self.data[:,1])
    
    def __getitem__(self, index):
        return self.t[index], self.x[index]
    
    def __len__(self):
        return self.len
    
    
batch_size = 10

dataset = VDPData()

# Lengths of the training and validation datasets
train_len = int(0.7*dataset.len)  #Train on 70% of the dataset
valid_len = dataset.len - train_len  #Validate on the rest

#Randomly split the data into training and validation datasets
train_data, valid_data = random_split(dataset, (train_len, valid_len))

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

n_hidden = 30
#Class representing the model (neural network)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(1, n_hidden)  #Input -> Hidden 1
        self.l2 = nn.Linear(n_hidden, n_hidden)  #Hidden 1 -> Hidden 2
        self.l3 = nn.Linear(n_hidden, 1)  #Hidden 2 -> Output
        
    def forward(self, x):
        activation_func = torch.sigmoid
        x = activation_func(self.l1(x))  #Apply first hidden layer and activation function
        x = activation_func(self.l2(x))  #Apply second hidden layer and activation function
        x = self.l3(x)  #Apply output layer
        return x #F.log_softmax(x)


NN = Net()  #Create an instance of the neural network


#Initialize weights and biases randomly
NN.l1.weight.data = torch.rand((n_hidden,1))
NN.l2.weight.data = torch.rand((n_hidden,n_hidden))
NN.l3.weight.data = torch.rand((1,n_hidden))

NN.l1.bias.data = torch.rand((1,n_hidden))
NN.l2.bias.data = torch.rand((1,n_hidden))
NN.l3.bias.data = torch.rand((1,1))


#Mean squared error loss
loss_func = nn.MSELoss()

#Stochastic gradient descent optimizer
optimizer = torch.optim.Adam(NN.parameters(), lr=0.004, weight_decay=1e-8)

plt.figure(1)
display_plot = True
lr_switched = False
    
#Training and validation loop
n_epochs = 2000
for epoch in range(n_epochs): #An epoch is a run of the entire training dataset
    
    train_loss, valid_loss = [], []    
    
    if epoch % (5*batch_size) == 0 and display_plot:
        #Plot true data
        plt.scatter(dataset.data[:,0], dataset.data[:,1], s=1, c='b')
        #plt.xlim(0,50)
        
        #Plot predicted data from this epoch
        predicted = NN(torch.from_numpy(dataset.data[:,0]).type(torch.float).unsqueeze(1))
        plt.scatter(dataset.data[:,0], predicted.detach().numpy(), s=1, c='r')
        plt.legend(('True','Predicted'), loc='upper left')
        plt.xlabel('t')
        plt.ylabel('x(t)')
        plt.show()
    
    #Training mode
    NN.train()
    for data in train_loader:
        
        #Split batch into inputs and outputs
        x, y = data[0].type(torch.float).unsqueeze(1), data[1].type(torch.float).unsqueeze(1)
        
        #Reset gradients to zero
        optimizer.zero_grad()
        
        #Forward propagation
        out = NN(x)
        
        #Loss computation
        loss = loss_func(out, y)
        
        #Backpropagation
        loss.backward()
        
        #Weight optimiation
        optimizer.step()  
        
        #Save training loss in this batch
        train_loss.append(loss.item())
        
    #Evaluation mode
    NN.eval()
    for data in valid_loader:
        
        #Split batch into inputs and outputs
        x, y = data[0].type(torch.float).unsqueeze(1), data[1].type(torch.float).unsqueeze(1)
        
        #Forward propagation
        out = NN(x)
        
        #Loss computation
        loss = loss_func(out, y)
        
        #Save validation loss in this batch
        valid_loss.append(loss.item())
    
    if epoch % 5 == 0:
        print("Epoch:", epoch, "\n  Mean Epoch Training Loss:  ", np.mean(train_loss),
              "\n  Mean Epoch Validation Loss:", np.mean(valid_loss))
    
    #Lower the learning rate once a decent approximation has been achieved
    if not lr_switched and np.mean(train_loss) < 0.45 :
        optimizer = torch.optim.Adam(NN.parameters(), lr=0.0015)
        print('\n\n\n\nLEARNING RATE SWITCHED FROM 0.004 TO 0.0015\n\n\n\n')
        lr_switched = True

    

plt.figure(2,figsize=(8,6))
plt.plot(dataset.data[:,0], dataset.data[:,1])
predicted = NN(torch.from_numpy(dataset.data[:,0]).type(torch.float).unsqueeze(1))
plt.plot(dataset.data[:,0], predicted.detach().numpy())
#plt.xlim(0,50)





