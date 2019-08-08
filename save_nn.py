import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(1) # reproducible


# create our data
x = torch.unsqueeze(torch.linspace(-1,1,100),dim = 1)
# unsqueeze change to 2D matix
y = x.pow(2) + 0.2*torch.rand(x.size())

x, y = Variable(x), Variable(y)

def save():
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,1)
    )
    optimizer = torch.optim.SGD(net1.parameters(), lr = 0.1)
    loss_fun = torch.nn.MSELoss()

    for t in range(200):
        prediction = net1(x)
        loss = loss_fun(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
    # plot results
    plt.figure(1,figsize=(10,3))
    plt.subplot(131)
    plt.title('Net1')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(),'r-',lw = 2)

    torch.save(net1,'net.pkl') #entire net 
    torch.save(net1.state_dict(),'net_params.pkl') # parameters
    

def restore_net():
    net2 = torch.load('net.pkl')
    prediction2 = net2(x)
    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction2.data.numpy(),'r-',lw = 2)


def restore_params():
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,1)
    )
    net3.load_state_dict(torch.load('net_params.pkl')) 
    prediction3 = net3(x)    
    plt.subplot(133)
    plt.title('Net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction3.data.numpy(),'r-',lw = 2)
    plt.show()

# save net1
save()

# restore entire net (may slow)
restore_net()

# restore only the net parameters
restore_params()