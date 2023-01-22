import numpy as np
import torch
import os


# create tensor
#w = torch.tensor(torch.rand([3,1]), requires_grad = True) # obsolete method
w = torch.rand([3,1]).clone().detach().requires_grad_(True)

# use adam optimizer with learning rate 0.1 to minimize the loss
opt = torch.optim.Adam([w], 0.1)

def model(x):
    # we define yhat to be our estimate y
    f = torch.stack([x * x, x, torch.ones_like(x)], 1)
    yhat = torch.squeeze(f @ w, 1)
    return yhat

def compute_loss(y, yhat):
    # loss = mean squared error
    loss = torch.nn.functional.mse_loss(yhat, y)
    return loss

def generate_data():
    # generate some training data based on a true function
    x = torch.rand(100) * 20 - 10
    y = 5 * x * x + 3
    return x, y

def train_step():
    x, y = generate_data()

    yhat = model(x)
    loss = compute_loss(y, yhat)

    opt.zero_grad()
    loss.backward()
    opt.step()

for _ in range(1000):
    train_step()

print(w.detach().numpy())