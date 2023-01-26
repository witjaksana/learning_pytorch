import torch
import time

x = torch.rand([500, 10])
z = torch.zeros([10])

#start = time.time()
#for i in range(500):
#    z += x[i]
#print(f"Took {time.time() - start} seconds.")

#start = time.time()
#for x_i in torch.unbind(x):
#    z += x_i
#print(f"Took {time.time() - start} seconds.")

start = time.time()
z = torch.sum(x, dim = 0)
print(f"Took {time.time() - start} seconds.")