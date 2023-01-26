import torch

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.rand(1))
        self.b = torch.nn.Parameter(torch.rand(1))

    def forward(self, x):
        yhat = self.a * x + self.b
        return yhat

class Net2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self, x):
        yhat = self.linear(x.unsqueeze(1)).squeeze(1)
        return yhat

def trial1():
    net = Net()
    x = torch.arange(100, dtype=torch.float32)
    y = net(x)
    for p in net.parameters():
        print(p)

def trial2():
    net = Net()
    x = torch.arange(100, dtype=torch.float32) / 100
    y = 5 * x + 3 + torch.rand(100) * 0.3

    # define a loss function and optimize
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    for i in range(10000):
        net.zero_grad()
        yhat = net(x)
        loss = criterion(yhat, y)
        loss.backward()
        optimizer.step()

    print(net.a)
    print(net.b)

def trial3():
    net = Net2()
    for p in net.parameters():
        print(p)



if __name__ == '__main__':
    #trial1()
    #trial2()
    trial3()