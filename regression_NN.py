# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 10:56:30 2019

@author: miwan
"""
# import libaries
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt  

# create our data
x = torch.unsqueeze(torch.linspace(-1,1,100),dim =1)
# unsqueeze change to 2D matix
y = x.pow(2) + 0.2*torch.rand(x.size())

x, y = Variable(x), Variable(y)

plt.scatter(x.numpy(),y.numpy())
plt.show()

class Net(torch.nn.Module):
    def __init__(self,n_features,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self,x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

net = Net(1,10,1)
print(net)

plt.ion()
plt.show()
optimizer = torch.optim.SGD(net.parameters(), lr = 0.5)
loss_func = torch.nn.MSELoss()

for t in range(100):
    prediction = net(x)
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',0.25)
#        plt.text(0.5,0, "Loss = %.4f" % loss.data[0])
        plt.pause(0.1)
    plt.ioff()
    plt.show()

