# import libaries
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt  

# create fake data
x = torch.unsqueeze(torch.linspace(-1,1,100),dim =1)
# unsqueeze change to 2D matix
y = x.pow(2) + 0.2*torch.rand(x.size())

x, y = Variable(x), Variable(y)

plt.scatter(x.numpy(),y.numpy())
plt.show()

