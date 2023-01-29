import numpy as np
import torch

x = np.float32(1)
y = np.float32(1e-50)

z = x * y / y
print(z)

print(np.nextafter(np.float32(0), np.float32(1)))
print(np.finfo(np.float32).max)

def unstable_softmax(logits):
    exp = torch.exp(logits)
    return exp / torch.sum(exp)

print(unstable_softmax(torch.tensor([1000., 0.])).numpy()) 
