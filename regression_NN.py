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
    def __init__(self,n_features,n_hidden,n_output)：
        super(Net,self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x
net = Net(1,10,1)
print(net)